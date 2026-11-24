"""Epoch-by-epoch comparison of student runs (GT only vs GT+KD): markdown table + small-multiples plot."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RUNS = [('student_none', 'GT only', '#eb6834'), ('student_full', 'GT + KD', '#2a78d6')]
RUNS += [(r, r.replace('student_', 'KD '), c) for r, c in [('student_logit', '#1baf7a'), ('student_feat', '#6250d6')]
         if os.path.exists(f'{ROOT}/runs/{r}/history.json')]
METRICS = [('mAP50_95', 'mAP@50:95'), ('mAP50', 'mAP@50'), ('mAP75', 'mAP@75'), ('F1', 'best F1 (IoU 0.5)')]
teacher = json.load(open(f'{ROOT}/runs/teacher_eval.json')) if os.path.exists(f'{ROOT}/runs/teacher_eval.json') else None
H = {r: json.load(open(f'{ROOT}/runs/{r}/history.json')) for r, _, _ in RUNS if os.path.exists(f'{ROOT}/runs/{r}/history.json')}

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for ax, (k, title) in zip(axes.flat, METRICS):
    for r, lab, col in RUNS:
        if r not in H: continue
        ep = [h['epoch'] for h in H[r]]; v = [h[k] for h in H[r]]
        ax.plot(ep, v, color=col, lw=2, label=lab)
        ax.annotate(f'{v[-1]:.3f}', (ep[-1], v[-1]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color='#52514e')
    if teacher:
        ax.axhline(teacher[k], color='#52514e', lw=1, ls='--', label='teacher YOLO11-L')
    ax.set_title(title, loc='left', fontsize=11, color='#0b0b0b')
    ax.grid(axis='y', color='#e5e4e0', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#b5b4ae')
    ax.tick_params(colors='#52514e', labelsize=8); ax.set_ylim(bottom=0)
for ax in axes[1]: ax.set_xlabel('epoch', color='#52514e')
axes[0, 0].legend(frameon=False, fontsize=9, loc='lower right')
fig.suptitle('Tiny student (0.63M params, conv/dwconv/maxpool/add/relu) - cyclist val set', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); fig.savefig(f'{ROOT}/results/curves.png', dpi=130)

# markdown table
base, kd = H.get('student_none', []), H.get('student_full', [])
lines = ['| epoch | mAP50-95 GT | mAP50-95 KD | Δ | mAP50 GT | mAP50 KD | Δ | F1 GT | F1 KD |', '|---|---|---|---|---|---|---|---|---|']
for b, d in zip(base, kd):
    lines.append(f"| {b['epoch']} | {b['mAP50_95']:.3f} | {d['mAP50_95']:.3f} | {d['mAP50_95']-b['mAP50_95']:+.3f} | "
                 f"{b['mAP50']:.3f} | {d['mAP50']:.3f} | {d['mAP50']-b['mAP50']:+.3f} | {b['F1']:.3f} | {d['F1']:.3f} |")
summary = []
for r, lab, _ in RUNS:
    if r in H:
        best = max(H[r], key=lambda h: h['mAP50_95'])
        summary.append(f"| {lab} | {H[r][-1]['mAP50_95']:.3f} | {H[r][-1]['mAP50']:.3f} | {best['mAP50_95']:.3f} (ep {best['epoch']}) | "
                       f"{sum(h['train_time_s'] for h in H[r])/len(H[r])/60:.1f} |")
if teacher:
    summary.append(f"| Teacher YOLO11-L (25.3M) | {teacher['mAP50_95']:.3f} | {teacher['mAP50']:.3f} | - | - |")
open(f'{ROOT}/results/comparison.md', 'w').write(
    '| run | final mAP50-95 | final mAP50 | best mAP50-95 | min/epoch |\n|---|---|---|---|---|\n' + '\n'.join(summary)
    + '\n\n### Epoch by epoch\n\n' + '\n'.join(lines) + '\n')
print(open(f'{ROOT}/results/comparison.md').read())
