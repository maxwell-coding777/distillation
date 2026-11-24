"""Tiny detector using ONLY: Conv, depthwise Conv, MaxPool, AvgPool, Add, ReLU.
(BatchNorm is used for training and folded into the convs at export -> no extra op at inference.)
No concat, no upsample, no sigmoid/SiLU, no elementwise multiply.

Output format is identical to a YOLOv8/11 Detect head (reg_max=16 DFL bins + nc class logits, strides 8/16/32),
so the student's dense predictions are anchor-aligned with the YOLO teacher -> direct dense logit distillation.
Box decoding (softmax over DFL bins) and the class sigmoid are post-processing, outside the network, as with any
NPU deployment of YOLO."""
import math
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        layers = [nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(c2)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2 block: pw expand -> dw 3x3 -> pw project (+ Add if same shape)."""

    def __init__(self, c1, c2, s=1, e=3):
        super().__init__()
        ch = c1 * e
        self.block = nn.Sequential(ConvBNReLU(c1, ch, 1), ConvBNReLU(ch, ch, 3, s, g=ch), ConvBNReLU(ch, c2, 1, act=False))
        self.res = s == 1 and c1 == c2

    def forward(self, x):
        return x + self.block(x) if self.res else self.block(x)


class SPPAdd(nn.Module):
    """SPPF-like receptive-field enlargement without concat: x + mp5(x) + mp5(mp5(x)) + mp5^3(x) then 1x1."""

    def __init__(self, c):
        super().__init__()
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.cv = ConvBNReLU(c, c, 1)

    def forward(self, x):
        y1 = self.pool(x); y2 = self.pool(y1); y3 = self.pool(y2)
        return self.cv(x + y1 + y2 + y3)


def stage(c1, c2, n, s):
    return nn.Sequential(InvertedResidual(c1, c2, s), *[InvertedResidual(c2, c2, 1) for _ in range(n - 1)])


class DWSep(nn.Sequential):
    def __init__(self, c1, c2, s=1):
        super().__init__(ConvBNReLU(c1, c1, 3, s, g=c1), ConvBNReLU(c1, c2, 1))


class StudentDet(nn.Module):
    def __init__(self, nc=1, reg_max=16, w=(16, 24, 48, 96, 160), neck=64):
        super().__init__()
        self.nc, self.reg_max, self.stride = nc, reg_max, torch.tensor([8.0, 16.0, 32.0])
        c0, c1, c2, c3, c4 = w
        self.stem = nn.Sequential(ConvBNReLU(3, c0, 3, 2), DWSep(c0, c0))      # /2
        self.s2 = stage(c0, c1, 2, 2)                                           # /4
        self.s3 = stage(c1, c2, 3, 2)                                           # /8
        self.s4 = stage(c2, c3, 4, 2)                                           # /16
        self.s5 = nn.Sequential(stage(c3, c4, 2, 2), SPPAdd(c4))                # /32
        # Bottom-up PAN with Add only (no upsampling / concat)
        self.lat3, self.lat4, self.lat5 = ConvBNReLU(c2, neck, 1), ConvBNReLU(c3, neck, 1), ConvBNReLU(c4, neck, 1)
        self.ctx3 = nn.Sequential(DWSep(neck, neck), DWSep(neck, neck))         # enlarge P3 receptive field
        self.down4, self.down5 = DWSep(neck, neck, 2), DWSep(neck, neck, 2)
        self.fuse4, self.fuse5 = DWSep(neck, neck), DWSep(neck, neck)
        # Decoupled YOLO-style head per level
        self.box = nn.ModuleList(nn.Sequential(DWSep(neck, neck), DWSep(neck, neck), nn.Conv2d(neck, 4 * reg_max, 1)) for _ in range(3))
        self.cls = nn.ModuleList(nn.Sequential(DWSep(neck, neck), DWSep(neck, neck), nn.Conv2d(neck, nc, 1)) for _ in range(3))
        self._init_bias()

    def _init_bias(self, imgsz=320):
        for b, c, s in zip(self.box, self.cls, self.stride):
            b[-1].bias.data[:] = 1.0
            c[-1].bias.data[:] = math.log(5 / self.nc / (imgsz / s) ** 2)

    def features(self, x):
        x = self.s2(self.stem(x)); c3 = self.s3(x); c4 = self.s4(c3); c5 = self.s5(c4)
        p3 = self.ctx3(self.lat3(c3))
        p4 = self.fuse4(self.lat4(c4) + self.down4(p3))
        p5 = self.fuse5(self.lat5(c5) + self.down5(p4))
        return [p3, p4, p5]

    def head(self, feats):
        return [(b(f), c(f)) for f, b, c in zip(feats, self.box, self.cls)]

    def forward(self, x, return_feats=False):
        feats = self.features(x)
        out = self.head(feats)
        return (out, feats) if return_feats else out


if __name__ == '__main__':
    m = StudentDet()
    print(f'params: {sum(p.numel() for p in m.parameters()) / 1e6:.3f} M')
    try:
        from thop import profile
        print(f'GFLOPs@320: {2 * profile(m, (torch.zeros(1, 3, 320, 320),), verbose=False)[0] / 1e9:.3f}')
    except Exception:
        pass
    print([tuple(t.shape) for o in m(torch.zeros(1, 3, 320, 320)) for t in o])
