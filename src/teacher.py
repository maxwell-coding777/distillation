"""Frozen ultralytics YOLO teacher exposing raw Detect outputs [(box, cls)] and neck features (P3, P4, P5)."""
import torch
import torch.nn as nn
from ultralytics import YOLO


class Teacher(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.model = YOLO(weights).model.float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.det = self.model.model[-1]
        self.channels = [self.det.cv2[i][0].conv.in_channels for i in range(self.det.nl)]
        self._feats = None
        self.det.register_forward_pre_hook(lambda m, inp: setattr(self, '_feats', list(inp[0])))

    def train(self, mode=True):  # always eval (frozen BN)
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        self.model(x)
        feats = self._feats
        preds = [(self.det.cv2[i](f), self.det.cv3[i](f)) for i, f in enumerate(feats)]
        return preds, feats
