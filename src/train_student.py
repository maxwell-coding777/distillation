"""Train the tiny student with ground truth only (--kd none) or GT + distillation (--kd full|logit|feat).
Both runs share seed, data order, augmentation, optimizer, schedule and EMA -> the KD terms are the only difference."""
import argparse, json, math, os, random, time, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import YoloDataset, collate
from student import StudentDet
from losses import DetLoss, KDLoss
from evaluate import evaluate

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ap = argparse.ArgumentParser()
ap.add_argument('--kd', default='none', choices=['none', 'full', 'logit', 'feat'])
ap.add_argument('--teacher', default=f'{ROOT}/runs/teacher/weights/best.pt')
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-3)
ap.add_argument('--imgsz', type=int, default=320)
ap.add_argument('--close_mosaic', type=int, default=5)
ap.add_argument('--w_cls', type=float, default=1.0)
ap.add_argument('--w_ld', type=float, default=0.5)
ap.add_argument('--w_feat', type=float, default=2.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--name', default=None)
ap.add_argument('--resume', action='store_true', help='resume from runs/<name>/ckpt.pt if present')
ap.add_argument('--limit', type=int, default=0, help='debug: limit batches per epoch')
a = ap.parse_args()
name = a.name or f'student_{a.kd}'
out = f'{ROOT}/runs/{name}'; os.makedirs(out, exist_ok=True)
torch.set_num_threads(a.threads)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

data = f'{ROOT}/data/cyclist'
tr = YoloDataset(data, 'train', a.imgsz, augment=True); va = YoloDataset(data, 'val', a.imgsz)
g = torch.Generator().manual_seed(a.seed)
tl = DataLoader(tr, a.bs, shuffle=True, num_workers=3, collate_fn=collate, generator=g, drop_last=True, persistent_workers=False,
                worker_init_fn=lambda w: (np.random.seed(torch.initial_seed() % 2 ** 32), random.seed(torch.initial_seed())))
vl = DataLoader(va, 32, num_workers=2, collate_fn=collate)

model = StudentDet()
ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
crit = DetLoss()
teacher = kd = None
if a.kd != 'none':
    from teacher import Teacher
    teacher = Teacher(a.teacher)
    w = dict(full=(a.w_cls, a.w_ld, a.w_feat), logit=(a.w_cls, a.w_ld, 0.0), feat=(0.0, 0.0, a.w_feat))[a.kd]
    kd = KDLoss([64, 64, 64], teacher.channels, w_cls=w[0], w_ld=w[1], w_feat=w[2])

params = list(model.parameters()) + (list(kd.parameters()) if kd else [])
decay = [p for p in params if p.ndim > 1]; no_decay = [p for p in params if p.ndim <= 1]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.05}, {'params': no_decay, 'weight_decay': 0}], lr=a.lr, betas=(0.9, 0.999))
nb = a.limit or len(tl); total = a.epochs * nb; warm = max(nb, 300)
lr_at = lambda it: a.lr * (it / warm if it < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - warm) / (total - warm))))
updates = 0
hist = []
start_ep = 0
ck = f'{out}/ckpt.pt'
if a.resume and os.path.exists(ck):
    c = torch.load(ck, map_location='cpu', weights_only=False)
    model.load_state_dict(c['model']); ema.load_state_dict(c['ema']); opt.load_state_dict(c['opt'])
    if kd is not None: kd.load_state_dict(c['kd'])
    start_ep, updates, hist = c['epoch'], c['updates'], c['hist']
    random.setstate(c['py_rng']); np.random.set_state(c['np_rng']); torch.set_rng_state(c['torch_rng'])
    g.manual_seed(a.seed + start_ep)
    print(f'resumed from epoch {start_ep}', flush=True)
print(f'{name}: student {sum(p.numel() for p in model.parameters())/1e6:.3f}M params | {len(tr)} train / {len(va)} val imgs | {nb} it/epoch', flush=True)
for ep in range(start_ep, a.epochs):
    if ep >= a.epochs - a.close_mosaic:
        tr.mosaic = 0.0
    model.train(); t0 = time.time(); agg = torch.zeros(6); n = 0
    for it, (imgs, targets) in enumerate(tl):
        if a.limit and it >= a.limit: break
        gi = ep * nb + it
        for pg in opt.param_groups: pg['lr'] = lr_at(gi)
        x = imgs.float() / 255
        s_preds, s_feats = model(x, return_feats=True)
        loss, items = crit(s_preds, targets)
        loss_kd_items = torch.zeros(3)
        if kd is not None:
            t_preds, t_feats = teacher(x)
            l_kd, loss_kd_items = kd(s_preds, t_preds, s_feats, t_feats)
            loss = loss + l_kd
        opt.zero_grad(set_to_none=True)
        (loss * a.bs).backward()  # YOLO convention: loss scaled by batch size
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        updates += 1; d = 0.9999 * (1 - math.exp(-updates / 2000))
        with torch.no_grad():
            for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                if pe.dtype.is_floating_point: pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                else: pe.copy_(pm)
        agg += torch.cat([items, loss_kd_items]); n += 1
        if it % 20 == 0:
            print(f'  ep {ep+1} it {it}/{nb} loss {loss.item():.3f} gt(box,cls,dfl)={items.numpy().round(3)} kd(cls,ld,feat)={loss_kd_items.numpy().round(3)}', flush=True)
    ttrain = time.time() - t0
    m = evaluate(ema, vl)
    rec = dict(epoch=ep + 1, **{k: round(v, 5) for k, v in m.items()}, train_time_s=round(ttrain, 1), lr=lr_at(gi),
               **dict(zip(['box', 'cls', 'dfl', 'kd_cls', 'kd_ld', 'kd_feat'], (agg / max(n, 1)).numpy().round(4).tolist())))
    hist.append(rec); print(json.dumps(rec), flush=True)
    json.dump(hist, open(f'{out}/history.json', 'w'), indent=1)
    torch.save(ema.state_dict(), f'{out}/last.pt')
    if m['mAP50_95'] >= max(h['mAP50_95'] for h in hist): torch.save(ema.state_dict(), f'{out}/best.pt')
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), kd=kd.state_dict() if kd else None,
                    epoch=ep + 1, updates=updates, hist=hist, py_rng=random.getstate(), np_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state()), ck + '.tmp')
    os.replace(ck + '.tmp', ck)
