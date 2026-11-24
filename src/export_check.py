"""Fuse BN into convs, export to ONNX and list operator types -> proves the deploy graph only uses
Conv (incl. depthwise = grouped Conv), MaxPool, AveragePool, Add, Relu."""
import collections, sys, torch, torch.nn as nn, onnx
from student import StudentDet

@torch.no_grad()
def fuse(model):
    """Fold every Conv2d -> BatchNorm2d pair (inside any nn.Sequential) into the Conv; BN becomes Identity."""
    for seq in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        mods = list(seq.children())
        for i in range(len(mods) - 1):
            conv, bn = mods[i], mods[i + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                s = bn.weight / (bn.running_var + bn.eps).sqrt()
                b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
                conv.weight.mul_(s[:, None, None, None])
                conv.bias = nn.Parameter(bn.bias + (b - bn.running_mean) * s)
                seq[i + 1] = nn.Identity()
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    return model

m = StudentDet().eval()
if len(sys.argv) > 1: m.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
x = torch.rand(1, 3, 320, 320)
ref = [tuple(t.clone() for t in o) for o in m(x)]
f = fuse(m).eval()
err = max((a - b).abs().max().item() for o, p in zip(ref, f(x)) for a, b in zip(o, p))
path = 'student.onnx'

torch.onnx.export(f, x, path, opset_version=13, dynamo=False,
                  output_names=[f'{k}_p{l}' for l in (3, 4, 5) for k in ('box', 'cls')])
mp = onnx.load(path); g = mp.graph
# Exporter emits Identity nodes that merely alias weight initializers (no compute): rewire & drop them
inits = {i.name for i in g.initializer}
alias = {n.output[0]: n.input[0] for n in g.node if n.op_type == 'Identity' and n.input[0] in inits}
for n in g.node:
    for k, i in enumerate(n.input):
        if i in alias: n.input[k] = alias[i]
keep = [n for n in g.node if not (n.op_type == 'Identity' and n.output[0] in alias)]
del g.node[:]; g.node.extend(keep); onnx.checker.check_model(mp); onnx.save(mp, path)
ops = collections.Counter(n.op_type for n in g.node)
convs = [n for n in g.node if n.op_type == 'Conv']
dw = sum(any(a.name == 'group' and a.i > 1 for a in n.attribute) for n in convs)
print(f'max |fused - unfused| = {err:.2e}')
print('ONNX ops:', dict(ops), f'(of which depthwise Conv: {dw})')
allowed = {'Conv', 'MaxPool', 'AveragePool', 'Add', 'Relu'}
print('ONLY allowed ops:', set(ops) <= allowed, '| forbidden:', set(ops) - allowed)
