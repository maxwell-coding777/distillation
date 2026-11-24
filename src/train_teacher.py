"""Fine-tune a large COCO-pretrained YOLO11 on the cyclist subset (teacher)."""
import argparse, os
from ultralytics import YOLO
ap = argparse.ArgumentParser(); ap.add_argument('--model', default='yolo11l.pt'); ap.add_argument('--epochs', type=int, default=15)
ap.add_argument('--imgsz', type=int, default=320); a = ap.parse_args()
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
YOLO(a.model).train(data=f'{root}/data/cyclist/cyclist.yaml', epochs=a.epochs, imgsz=a.imgsz, batch=16, device='cpu',
                    workers=4, cache='ram', close_mosaic=3, project=f'{root}/runs', name='teacher', exist_ok=True,
                    single_cls=True, seed=0, plots=False)
