# Which clean corpus clips the filter reader takes for filtered (kinds only, never names), and what curve it reads there.
import sys, numpy as np, torch, json
from common import *
import ffeat, ftest
net = ftest.load_net(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32); C = np.load('data/corpus_E.npz'); meta = json.load(open(CORPUS + 'meta.json'))
pts = [100, 200, 400, 800, 1600, 3200, 6400, 12800]; ib = [int(np.argmin(np.abs(ffeat.FC - p))) for p in pts]; rows = []
for e, n in zip(C['E'], C['names']):
    L, t = ftest.read(net, e); X, ref = ffeat.curves(e.astype(np.float64)); rows.append((t, kind(str(n)), meta[str(n).replace('.wav', '')]['split'], L, X[0]))
rows.sort(key=lambda r: -r[0])
for t, k, sp, L, S in rows[:int(sys.argv[3]) if len(sys.argv) > 3 else 20]:
    print('sure %+5.1f %-4s %-4s read %s | the sound itself %s' % (t, k, sp, ' '.join('%+4.0f' % L[i] for i in ib), ' '.join('%+4.0f' % S[i] for i in ib)))
import collections
for thr in (0, 2, 4): print('above', thr, dict(collections.Counter(k for t, k, sp, L, S in rows if t > thr)))
