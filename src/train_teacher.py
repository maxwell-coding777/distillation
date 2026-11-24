"""Fine-tune a large COCO-pretrained YOLO11 on the cyclist subset (teacher)."""
import argparse, os
from ultralytics import YOLO
ap = argparse.ArgumentParser(); ap.add_argument('--model', default='yolo11l.pt'); ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--imgsz', type=int, default=320); a = ap.parse_args()
last = None
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
last = f'{root}/runs/teacher/weights/last.pt'
if os.path.exists(last):  # resume after a container restart
    YOLO(last).train(resume=True)
    raise SystemExit
YOLO(a.model).train(data=f'{root}/data/cyclist/cyclist.yaml', epochs=a.epochs, imgsz=a.imgsz, batch=16, device='cpu',
                    workers=4, cache='ram', close_mosaic=2, project=f'{root}/runs', name='teacher', exist_ok=True,
                    single_cls=True, seed=0, plots=False,
                    # explicit fine-tuning recipe: optimizer='auto' picks AdamW with warmup_bias_lr=0.1, which
                    # wrecks the pretrained BN biases (train loss went UP, val cls loss 4245 at epoch 1)
                    optimizer='SGD', lr0=0.005, lrf=0.05, momentum=0.937, warmup_epochs=1, warmup_bias_lr=0.0, cos_lr=True)
