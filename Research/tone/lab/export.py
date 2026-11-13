# Checkpoint of the trained net -> tone_model.npz next to tone.py (plain float32 arrays, read by tone.py without torch).
import sys, numpy as np, torch
tag = sys.argv[1]; restraint = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
ck = torch.load('work/net_%s.pt' % tag, map_location='cpu'); st = ck['state']; out = dict(inp_w=st['inp.weight'][:, :, 0].numpy(), inp_b=st['inp.bias'].numpy(), out_w=st['out.weight'][:, :, 0].numpy(), out_b=st['out.bias'].numpy())
n = len([k for k in st if k.startswith('convs.') and k.endswith('.weight')]); dil = (1, 2, 4, 8, 16, 8, 4, 2, 1)[:n]
for i in range(n): out['c%d_w' % i] = st['convs.%d.weight' % i].numpy(); out['c%d_b' % i] = st['convs.%d.bias' % i].numpy()
out['dil'] = np.array(dil, np.int32); out['restraint'] = np.float32(restraint)
np.savez('/tmp/claude-0/v2/tone/tone_model.npz', **{k: (v.astype(np.float32) if v.dtype.kind == 'f' else v) for k, v in out.items()})
print('weights:', sum(v.size for k, v in out.items() if k not in ('dil', 'restraint')), 'config:', ck['cfg'])
