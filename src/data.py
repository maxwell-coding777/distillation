"""Minimal YOLO-format dataset with mosaic + random affine + HSV + flip (RAM cached).
The SAME augmented batch is fed to student and teacher during distillation."""
import glob, math, os, random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YoloDataset(Dataset):
    def __init__(self, root, split, imgsz=320, augment=False, mosaic=1.0):
        self.files = sorted(glob.glob(f'{root}/images/{split}/*.jpg'))
        self.imgsz, self.augment, self.mosaic = imgsz, augment, mosaic
        self.imgs, self.labels = [], []
        for f in self.files:
            im = cv2.imread(f)
            h, w = im.shape[:2]; r = imgsz / max(h, w)
            self.imgs.append(cv2.resize(im, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA))
            lf = f.replace('/images/', '/labels/')[:-4] + '.txt'
            lb = np.loadtxt(lf, ndmin=2).reshape(-1, 5) if os.path.getsize(lf) else np.zeros((0, 5))
            self.labels.append(lb.astype(np.float32))  # cls, xc, yc, w, h normalised

    def __len__(self):
        return len(self.files)

    def _xyxy(self, i, w, h, padx=0, pady=0):
        lb = self.labels[i].copy(); b = np.zeros((len(lb), 4), np.float32)
        b[:, 0] = (lb[:, 1] - lb[:, 3] / 2) * w + padx; b[:, 1] = (lb[:, 2] - lb[:, 4] / 2) * h + pady
        b[:, 2] = (lb[:, 1] + lb[:, 3] / 2) * w + padx; b[:, 3] = (lb[:, 2] + lb[:, 4] / 2) * h + pady
        return b

    def _letterbox(self, i):
        im = self.imgs[i]; h, w = im.shape[:2]; s = self.imgsz
        out = np.full((s, s, 3), 114, np.uint8); px, py = (s - w) // 2, (s - h) // 2
        out[py:py + h, px:px + w] = im
        return out, self._xyxy(i, w, h, px, py)

    def _mosaic(self, i):
        s = self.imgsz; xc, yc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        canvas = np.full((2 * s, 2 * s, 3), 114, np.uint8); boxes = []
        for k, j in enumerate([i] + random.choices(range(len(self)), k=3)):
            im = self.imgs[j]; h, w = im.shape[:2]
            if k == 0: x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc; x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            elif k == 1: x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc; x1b, y1b = 0, h - (y2a - y1a)
            elif k == 2: x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h); x1b, y1b = w - (x2a - x1a), 0
            else: x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h); x1b, y1b = 0, 0
            canvas[y1a:y2a, x1a:x2a] = im[y1b:y1b + (y2a - y1a), x1b:x1b + (x2a - x1a)]
            boxes.append(self._xyxy(j, w, h, x1a - x1b, y1a - y1b))
        return canvas, np.concatenate(boxes), -s // 2

    def _affine(self, im, boxes, border):
        s = self.imgsz; H, W = im.shape[:2]
        C = np.eye(3); C[0, 2], C[1, 2] = -W / 2, -H / 2
        R = np.eye(3); sc = random.uniform(0.5, 1.5); R[:2] = cv2.getRotationMatrix2D((0, 0), 0, sc)
        T = np.eye(3); T[0, 2] = random.uniform(0.4, 0.6) * s; T[1, 2] = random.uniform(0.4, 0.6) * s
        M = T @ R @ C
        im = cv2.warpAffine(im, M[:2], (s, s), borderValue=(114, 114, 114))
        if len(boxes):
            pts = np.ones((len(boxes) * 4, 3)); pts[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)
            pts = (pts @ M.T)[:, :2].reshape(-1, 8)
            nb = np.stack([pts[:, 0::2].min(1), pts[:, 1::2].min(1), pts[:, 0::2].max(1), pts[:, 1::2].max(1)], 1).clip(0, s)
            ow, oh = (boxes[:, 2] - boxes[:, 0]) * sc, (boxes[:, 3] - boxes[:, 1]) * sc
            nw, nh = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
            keep = (nw > 2) & (nh > 2) & (nw * nh / (ow * oh + 1e-9) > 0.1)   # drop boxes mostly cropped out
            boxes = nb[keep]
        return im, boxes

    @staticmethod
    def _hsv(im, hg=0.015, sg=0.7, vg=0.4):
        r = np.random.uniform(-1, 1, 3) * [hg, sg, vg] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)); x = np.arange(256)
        lut = [((x * r[0]) % 180).astype(np.uint8), np.clip(x * r[1], 0, 255).astype(np.uint8), np.clip(x * r[2], 0, 255).astype(np.uint8)]
        return cv2.cvtColor(cv2.merge((cv2.LUT(hue, lut[0]), cv2.LUT(sat, lut[1]), cv2.LUT(val, lut[2]))), cv2.COLOR_HSV2BGR)

    def __getitem__(self, i):
        if self.augment:
            if random.random() < self.mosaic: im, boxes, border = self._mosaic(i)
            else: im, boxes = self._letterbox(i); border = 0
            im, boxes = self._affine(im, boxes, border)
            im = self._hsv(im)
            if random.random() < 0.5:
                im = np.ascontiguousarray(im[:, ::-1]); boxes = boxes.copy(); boxes[:, [0, 2]] = self.imgsz - boxes[:, [2, 0]]
        else:
            im, boxes = self._letterbox(i)
        im = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)))  # BGR->RGB, CHW uint8
        return im, torch.from_numpy(np.asarray(boxes, np.float32).reshape(-1, 4))


def collate(batch):
    ims, boxes = zip(*batch)
    tg = [torch.cat([torch.full((len(b), 1), i, dtype=torch.float32), b], 1) for i, b in enumerate(boxes)]
    return torch.stack(ims), torch.cat(tg)  # targets: [N, 5] = (batch_idx, x1, y1, x2, y2) in pixels
