"""Evaluate the teacher with the SAME evaluator/letterbox as the students (+ ultralytics val as a cross-check)."""
import json, os, torch
from torch.utils.data import DataLoader
from data import YoloDataset, collate
from teacher import Teacher
from evaluate import evaluate
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
w = f'{ROOT}/runs/teacher/weights/best.pt'
t = Teacher(w)
vl = DataLoader(YoloDataset(f'{ROOT}/data/cyclist', 'val', 320), 16, num_workers=2, collate_fn=collate)
m = evaluate(t, vl, fwd=lambda mod, x: mod(x)[0])
m['params_M'] = sum(p.numel() for p in t.parameters()) / 1e6
print('our evaluator:', m)
json.dump(m, open(f'{ROOT}/runs/teacher_eval.json', 'w'), indent=1)
from ultralytics import YOLO
r = YOLO(w).val(data=f'{ROOT}/data/cyclist/cyclist.yaml', imgsz=320, batch=16, device='cpu', rect=False, plots=False, verbose=False)
print('ultralytics val: mAP50 %.4f mAP50-95 %.4f' % (r.box.map50, r.box.map))
