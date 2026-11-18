# What the picture's own resolution allows: the TRUE moving tone curve of each bench case, read with the probe signal through
# the same outside effect with the same settings (96 bands, 21 ms frames), taken off the wet sound and put on the other sound.
#   python3 oracle.py family [lift dB]
import sys, numpy as np
sys.path[:0] = ['/tmp/claude-0/v2/moving', '/tmp/claude-0/v2/sweep', '/tmp/claude-0/v2/detect/py']
import pic, zoo_fx as Z
from lab import cases, load, score
fam = sys.argv[1]; LIFT = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0; rows = []
def truth(c, n):
    p = pic.probe(n); ps = np.stack([p, p], 1); w = Z.apply(c['meta']['effect'], ps, c['meta']['params']); return [pic.curve(p, w[:n, ch]) for ch in range(2)]
for d in cases(fam):
    cA = load(d, 'A'); cB = load(d, 'B'); n = len(cA['wet']); G = truth(cA, n)
    for c, o, tag in ((cA, cB, 'A'), (cB, cA, 'B')):
        out = np.stack([pic.render(c['wet'][:, ch], G[ch], -1.0, LIFT) for ch in range(2)], 1); onto = np.stack([pic.render(o['dry'][:, ch], G[ch], 1.0) for ch in range(2)], 1)
        r = score.removed(c['dry'], c['wet'], out)[0]; ro = score.removed(o['wet'], o['dry'], onto)[0]; rows.append((r, ro))
        print('%-20s %s out %7.1f onto %7.1f' % (c['name'], tag, r, ro), flush=True)
v = np.array(rows); print('%s: probe-truth ceiling out: median %.1f mean %.1f worse %d/%d | onto: median %.1f mean %.1f worse %d/%d' % (fam, np.median(v[:, 0]), v[:, 0].mean(), (v[:, 0] < 0).sum(), len(v), np.median(v[:, 1]), v[:, 1].mean(), (v[:, 1] < 0).sum(), len(v)))
