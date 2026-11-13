# Does the training-time simulation (dry statistics + shifts computed from the curve) match what the statistics of really filtered sound do?
import numpy as np
from base import *
import feat, curves
z = np.load(WORK + 'bench_feats.npz'); names = list(z['names']); eff = z['effects']; sp = z['splits']
acc = {}
for d in case_dirs():
    c = load(d, 'A'); m = c['meta']
    if m['split'] != 'dev': continue
    i = names.index(c['name']); f, db = true_db(c)
    for o in range(1):
        v = np.concatenate([np.interp(curves.DENSE, f, db[o]), db[o][::NT // feat.N]]); df = f[1]; e = feat._e
        lo = np.clip(np.ceil(e[:-1] / df).astype(int), 0, len(f)); hi = np.clip(np.ceil(e[1:] / df).astype(int), 0, len(f)); w = hi - lo >= 2; cs = np.concatenate([[0.0], np.cumsum(db[o])]); T = np.interp(feat.GRID, f, db[o]); T[w] = (cs[hi[w]] - cs[lo[w]]) / (hi[w] - lo[w])
        pk = curves.pack(v, T); Fw = z['Aw'][i, o]; Fd = z['Ad'][i, o]; lev = Fd[0] + Fd[feat.NAMES.index('top')]; aud = lev > lev.max() - 60
        real = [Fw[0] - Fd[0], (Fw[0] + Fw[1]) - (Fd[0] + Fd[1]), (Fw[0] + Fw[2]) - (Fd[0] + Fd[2])]; row = []
        for k in range(3):
            a = real[k] - pk[k + 1]; a = a - a[aud].mean(); b = real[k] - T; b = b - b[aud].mean(); row += [np.sqrt((a[aud] ** 2).mean()), np.sqrt((b[aud] ** 2).mean())]
        acc.setdefault(m['effect'], []).append(row)
print('rms dB over audible cells: real shift of the statistic minus [simulated shift | plain exact curve], for lmean, floor, env')
for e, v in acc.items(): v = np.mean(v, 0); print('%-20s lmean %.2f | %.2f   floor %.2f | %.2f   env %.2f | %.2f' % (e, *v))
