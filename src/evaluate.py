"""Single-class detection evaluation (COCO-style mAP50 / mAP50-95, 101-pt interpolation) + P/R at best F1.
Works for any model returning [(box[B,4R,H,W], cls[B,nc,H,W])] per stride level (student or YOLO teacher)."""
import numpy as np
import torch
import torchvision
from ultralytics.utils.tal import make_anchors, dist2bbox
from losses import flatten

IOUS = np.linspace(0.5, 0.95, 10)


@torch.no_grad()
def postprocess(preds, strides=(8, 16, 32), reg_max=16, conf=0.001, iou=0.7, max_det=300):
    box, cls = flatten(preds)
    anchors, st = make_anchors([b for b, _ in preds], torch.tensor(strides, dtype=torch.float32), 0.5)
    B, A, _ = box.shape
    d = box.view(B, A, 4, reg_max).softmax(3) @ torch.arange(reg_max, dtype=torch.float32)
    xyxy = dist2bbox(d, anchors, xywh=False) * st
    scores = cls.sigmoid().amax(-1)
    out = []
    for i in range(B):
        k = scores[i] > conf
        b, s = xyxy[i][k], scores[i][k]
        keep = torchvision.ops.nms(b, s, iou)[:max_det]
        out.append(torch.cat([b[keep], s[keep, None]], 1))
    return out


def ap101(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapezoid(np.interp(x, mrec, mpre), x)


@torch.no_grad()
def evaluate(model, loader, fwd=None):
    """fwd(model, imgs_float) -> preds; defaults to model(imgs)."""
    fwd = fwd or (lambda m, x: m(x))
    stats, n_gt = [], 0
    for imgs, targets in loader:
        dets = postprocess(fwd(model, imgs.float() / 255))
        for i, d in enumerate(dets):
            gt = targets[targets[:, 0] == i, 1:]
            n_gt += len(gt)
            correct = np.zeros((len(d), len(IOUS)), bool)
            if len(d) and len(gt):
                iou = torchvision.ops.box_iou(gt, d[:, :4]).numpy()
                for j, th in enumerate(IOUS):   # ultralytics-style greedy matching per threshold
                    x = np.argwhere(iou >= th)
                    if len(x):
                        m = np.concatenate([x, iou[x[:, 0], x[:, 1]][:, None]], 1)
                        m = m[m[:, 2].argsort()[::-1]]
                        m = m[np.unique(m[:, 1], return_index=True)[1]]
                        m = m[m[:, 2].argsort()[::-1]]
                        m = m[np.unique(m[:, 0], return_index=True)[1]]
                        correct[m[:, 1].astype(int), j] = True
            stats.append((correct, d[:, 4].numpy()))
    correct = np.concatenate([s[0] for s in stats]); conf = np.concatenate([s[1] for s in stats])
    order = np.argsort(-conf); correct, conf = correct[order], conf[order]
    tpc = correct.cumsum(0); fpc = (1 - correct).cumsum(0)
    rec = tpc / max(n_gt, 1); prec = tpc / np.maximum(tpc + fpc, 1e-9)
    aps = np.array([ap101(rec[:, j], prec[:, j]) for j in range(len(IOUS))]) if len(conf) else np.zeros(len(IOUS))
    f1 = 2 * prec[:, 0] * rec[:, 0] / np.maximum(prec[:, 0] + rec[:, 0], 1e-9) if len(conf) else np.zeros(1)
    b = int(f1.argmax())
    return dict(mAP50=float(aps[0]), mAP50_95=float(aps.mean()), mAP75=float(aps[5]),
                P=float(prec[b, 0]) if len(conf) else 0.0, R=float(rec[b, 0]) if len(conf) else 0.0, F1=float(f1[b]))
