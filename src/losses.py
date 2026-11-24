"""Ground-truth detection loss (YOLOv8/11 recipe) + distillation losses.

KD recipe (dense, anchor-aligned since student/teacher share strides 8/16/32 and the DFL head format):
  1. Classification KD  - BCE with the teacher's (temperature-softened) sigmoid scores as soft targets, modulated
     by |p_t - p_s|^gamma (quality-focal style, cf. BCKD ICCV'23 / QFL) so the ~2k easy-background anchors
     don't drown the signal; normalised by the teacher's foreground mass.
  2. Localization Distillation (LD, Zheng et al. CVPR'22 / TPAMI'23) - KL between the teacher's and student's
     DFL bin distributions (temperature T), weighted by the teacher's score -> distil "where" knowledge
     (box ambiguity), which GT one-hot DFL targets cannot provide.
  3. Feature KD - PKD (Cao et al. NeurIPS'22): MSE between per-channel standardised (Pearson) teacher and
     adapted student neck features. Insensitive to magnitude gaps between heterogeneous teacher/student.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox, bbox2dist, bbox_iou


def flatten(preds):
    """[(box[B,4R,H,W], cls[B,nc,H,W]) x levels] -> box [B, A, 4R], cls [B, A, nc]."""
    box = torch.cat([b.flatten(2) for b, _ in preds], 2).permute(0, 2, 1)
    cls = torch.cat([c.flatten(2) for _, c in preds], 2).permute(0, 2, 1)
    return box.contiguous(), cls.contiguous()


class DetLoss:
    def __init__(self, nc=1, reg_max=16, strides=(8, 16, 32), box=7.5, cls=0.5, dfl=1.5):
        self.nc, self.R, self.strides = nc, reg_max, torch.tensor(strides, dtype=torch.float32)
        self.gain = (box, cls, dfl)
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
        self.proj = torch.arange(reg_max, dtype=torch.float32)

    def decode(self, anchors, box):
        b, a, _ = box.shape
        d = box.view(b, a, 4, self.R).softmax(3).matmul(self.proj)
        return dist2bbox(d, anchors, xywh=False)

    def __call__(self, preds, targets):
        feats = [b for b, _ in preds]
        box, cls = flatten(preds)
        B = box.shape[0]
        anchors, stride_t = make_anchors(feats, self.strides, 0.5)
        # targets [N, 5] (bidx, x1,y1,x2,y2 px) -> padded [B, M, 4]
        counts = torch.bincount(targets[:, 0].long(), minlength=B) if len(targets) else torch.zeros(B, dtype=torch.long)
        M = max(int(counts.max()), 1)
        gt_b = torch.zeros(B, M, 4); gt_l = torch.zeros(B, M, 1)
        for i in range(B):
            t = targets[targets[:, 0] == i, 1:]
            gt_b[i, :len(t)] = t
        mask = gt_b.sum(2, keepdim=True) > 0
        pb = self.decode(anchors, box)
        _, tb, ts, fg, _ = self.assigner(cls.detach().sigmoid(), (pb.detach() * stride_t), anchors * stride_t, gt_l, gt_b, mask)
        tss = ts.sum().clamp(min=1)
        l_cls = F.binary_cross_entropy_with_logits(cls, ts, reduction='none').sum() / tss
        if fg.any():
            w = ts[fg].sum(-1, keepdim=True)
            tb_s = tb / stride_t
            l_box = ((1 - bbox_iou(pb[fg], tb_s[fg], xywh=False, CIoU=True)) * w).sum() / tss
            tl = bbox2dist(anchors, tb_s, self.R - 1)[fg].clamp(0, self.R - 1 - 0.01)
            logp = F.log_softmax(box[fg].view(-1, self.R), 1); t = tl.view(-1)
            lo = t.long(); wl = lo + 1 - t
            dfl = -(logp.gather(1, lo[:, None])[:, 0] * wl + logp.gather(1, (lo + 1)[:, None])[:, 0] * (1 - wl))
            l_dfl = (dfl.view(-1, 4).mean(1, keepdim=True) * w).sum() / tss
        else:
            l_box = l_dfl = box.sum() * 0
        g = self.gain
        return l_box * g[0] + l_cls * g[1] + l_dfl * g[2], torch.stack([l_box, l_cls, l_dfl]).detach()


class KDLoss(nn.Module):
    def __init__(self, s_ch, t_ch, reg_max=16, T_cls=1.0, T_ld=2.0, gamma=1.0, w_cls=1.0, w_ld=0.5, w_feat=2.0):
        super().__init__()
        self.R, self.T_cls, self.T_ld, self.gamma = reg_max, T_cls, T_ld, gamma
        self.w = (w_cls, w_ld, w_feat)
        # training-only 1x1 adapters (discarded at deploy)
        self.adapt = nn.ModuleList(nn.Conv2d(s, t, 1) for s, t in zip(s_ch, t_ch))

    @staticmethod
    def _std(x):  # per-sample, per-channel standardisation (Pearson)
        m = x.mean((2, 3), keepdim=True); s = x.std((2, 3), keepdim=True)
        return (x - m) / (s + 1e-6)

    def forward(self, s_preds, t_preds, s_feats, t_feats):
        sb, sc = flatten(s_preds); tb, tc = flatten(t_preds)
        if tc.shape[-1] != sc.shape[-1]:  # (debug with a non-finetuned COCO teacher)
            tc = tc.amax(-1, keepdim=True)
        # 1. classification KD
        pt = (tc / self.T_cls).sigmoid(); ps = (sc / self.T_cls).sigmoid()
        mod = (pt - ps).abs().pow(self.gamma).detach()
        norm = pt.sum().clamp(min=1)
        l_cls = (F.binary_cross_entropy_with_logits(sc / self.T_cls, pt, reduction='none') * mod).sum() / norm * self.T_cls ** 2
        # 2. localization distillation (KL on DFL bins), weighted by teacher confidence
        w = tc.sigmoid().amax(-1).detach()                                  # [B, A]
        B, A, _ = sb.shape
        ls = F.log_softmax(sb.view(B, A, 4, self.R) / self.T_ld, -1)
        lt = F.log_softmax(tb.view(B, A, 4, self.R) / self.T_ld, -1)
        kl = (lt.exp() * (lt - ls)).sum(-1).mean(-1)                        # [B, A]
        l_ld = (kl * w).sum() / w.sum().clamp(min=1) * self.T_ld ** 2
        # 3. PKD feature distillation
        l_feat = sum(F.mse_loss(self._std(a(s)), self._std(t)) / 2 for a, s, t in zip(self.adapt, s_feats, t_feats)) / len(s_feats)
        parts = torch.stack([l_cls, l_ld, l_feat])
        return (parts * torch.tensor(self.w)).sum(), parts.detach()
