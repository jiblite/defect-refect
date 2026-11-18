# Does reading again help? Read the curve, take it off, read what is left, add it to the curve, and so on. Per pass: the
# curve's size, the scores of taking the summed curve off and of putting it on the case's other sound, what the model says after.
#   python3 iter.py model passes family ...
import sys, os, json, numpy as np, torch
sys.path[:0] = ['/tmp/claude-0/v2/moving', '/tmp/claude-0/v2/sweep']
import moving, pic
from lab import cases, load, score
torch.set_num_threads(int(os.environ.get('THREADS', '1')))
model = moving.model_of('/tmp/claude-0/v2/moving/%s.pt' % sys.argv[1]); K = int(sys.argv[2]); rows = []
def reading(x):
    same = np.array_equal(x[:, 0], x[:, 1]); r = [moving.read(model, x[:, 0])]; r.append(r[0] if same else moving.read(model, x[:, 1]))
    return np.stack([g for g, _ in r]), np.maximum(1 - r[0][1][:, 0], 1 - r[1][1][:, 0])
for fam in sys.argv[3:]:
    for d in cases(fam):
        for w in 'AB':
            c = load(d, w); o = load(d, 'B' if w == 'A' else 'A'); x = c['wet']; F = moving.find(model, x)
            if not F['found']: print('%-20s %s not found' % (c['name'], w), flush=True); continue
            gate = F['gate']; m = gate > 0; Gt = F['G'].copy(); row = dict(fam=fam, name=c['name'] + '_' + w, kind=F['kind'], sure=F['sure'], passes=[]); txt = []
            for k in range(K):
                out = np.stack([pic.render(x[:, ch], Gt[ch], -1.0) for ch in range(2)], 1); onto = np.stack([pic.render(o['dry'][:, ch], Gt[ch], 1.0) for ch in range(2)], 1)
                sc = score.removed(c['dry'], x, out)[0]; so = score.removed(o['wet'], o['dry'], onto)[0]; G2, on2 = reading(out); T = min(len(m), len(on2)); say2 = float(on2[:T][m[:T]].mean())
                size = float(np.sqrt((Gt[:, m] ** 2).mean())); row['passes'].append(dict(size=size, out=sc, onto=so, after=say2)); txt.append('%5.2f dB %6.1f %6.1f (%.2f)' % (size, sc, so, say2))
                Gt[:, :T] += G2[:, :T] * gate[None, :T, None]
            rows.append(row); print('%-20s %s %-11s %.2f | %s' % (c['name'], w, F['kind'], F['sure'], ' | '.join(txt)), flush=True)
json.dump(rows, open('iter_%s.json' % sys.argv[1], 'w'))
