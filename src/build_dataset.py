"""Build a COCO-derived 'cyclist' subset in YOLO format.
cyclist = union box of a (person, bicycle) pair where the person rides the bike:
  - inter(person, bike) / area(person) >= 0.2
  - person horizontal center within the bike x-range
  - person top above bike top
Greedy 1-1 matching by overlap. Unmatched bikes (parked) & pedestrians = background (hard negatives)."""
import json, random, os, sys, concurrent.futures as cf, urllib.request
random.seed(0)
ROOT = os.environ.get('KD_DATA', os.path.join(os.path.dirname(__file__), '..', 'data'))
R = f'{ROOT}/raw/annotations'; OUT = f'{ROOT}/cyclist'

def cyclists(anns, cat_person=1, cat_bike=2):
    P = [a['bbox'] for a in anns if a['category_id'] == cat_person and not a['iscrowd']]
    B = [a['bbox'] for a in anns if a['category_id'] == cat_bike and not a['iscrowd']]
    pairs = []
    for i, (px, py, pw, ph) in enumerate(P):
        for j, (bx, by, bw, bh) in enumerate(B):
            iw = max(0, min(px+pw, bx+bw) - max(px, bx)); ih = max(0, min(py+ph, by+bh) - max(py, by))
            ov = iw*ih / max(pw*ph, 1e-6)
            if ov >= 0.2 and bx <= px+pw/2 <= bx+bw and py < by:
                pairs.append((ov, i, j))
    pairs.sort(reverse=True); up, ub, out = set(), set(), []
    for ov, i, j in pairs:
        if i in up or j in ub: continue
        up.add(i); ub.add(j); p, b = P[i], B[j]
        x1, y1 = min(p[0], b[0]), min(p[1], b[1]); x2, y2 = max(p[0]+p[2], b[0]+b[2]), max(p[1]+p[3], b[1]+b[3])
        out.append((x1, y1, x2, y2))
    has_bike, has_person = len(B) > 0, len(P) > 0
    return out, has_bike, has_person

def load(split):
    d = json.load(open(f'{R}/instances_{split}.json'))
    imgs = {i['id']: i for i in d['images']}; by = {}
    for a in d['annotations']: by.setdefault(a['image_id'], []).append(a)
    pos, neg_hard, neg = [], [], []
    for iid, im in imgs.items():
        cyc, hb, hp = cyclists(by.get(iid, []))
        rec = (im, cyc)
        if cyc: pos.append(rec)
        elif hb or hp: neg_hard.append(rec)   # parked bikes / pedestrians
        else: neg.append(rec)
    return pos, neg_hard, neg

def write(split, recs, coco_split):
    os.makedirs(f'{OUT}/images/{split}', exist_ok=True); os.makedirs(f'{OUT}/labels/{split}', exist_ok=True)
    def job(rec):
        im, cyc = rec; fn = im['file_name']; W, H = im['width'], im['height']
        dst = f'{OUT}/images/{split}/{fn}'
        if not os.path.exists(dst):
            for t in range(4):
                try: urllib.request.urlretrieve(f'http://images.cocodataset.org/{coco_split}/{fn}', dst); break
                except Exception: pass
        with open(f'{OUT}/labels/{split}/{fn[:-4]}.txt', 'w') as f:
            for x1, y1, x2, y2 in cyc:
                f.write(f'0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n')
    with cf.ThreadPoolExecutor(32) as ex: list(ex.map(job, recs))

tp, th, tn = load('train2017'); vp, vh, vn = load('val2017')
print('train pos/hardneg/neg', len(tp), len(th), len(tn), '| val', len(vp), len(vh), len(vn))
random.shuffle(tp); random.shuffle(th)
N_TRAIN_POS, N_VAL_POS_FROM_TRAIN = 2400, 300
val_from_train = tp[:N_VAL_POS_FROM_TRAIN]; train_pos = tp[N_VAL_POS_FROM_TRAIN:N_VAL_POS_FROM_TRAIN+N_TRAIN_POS]
train = train_pos + th[:1000]                     # ~30% hard negatives
val = vp + val_from_train + random.sample(vh, 250)
print('train', len(train), 'val', len(val), 'val cyclists', sum(len(c) for _, c in val))
write('train', train, 'train2017')
write('val', [r for r in val if r in vp or r in vh], 'val2017')
write('val', val_from_train, 'train2017')
open(f'{OUT}/cyclist.yaml', 'w').write(f'path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\nnames:\n  0: cyclist\n')
