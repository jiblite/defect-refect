# Classic shrinkage fitted for one curve type at a time (plus clean sounds): is there a type where thresholded detail of the floor statistic pays?
import sys, numpy as np
import feat, curves, model
from classic import *
F = np.load(WORK + 'all_feats.npy')[:1057].astype(np.float32); perm = np.random.default_rng(123).permutation(len(F)); tr = perm[100:]
pool, ty, names = curves.pool(20000, 1)
for t, nm in enumerate(names):
    sub = pool[ty == t]; X, C, k = sample(F[tr], sub, 4000, 0, null=0.5); W = model.weights(X); nul = np.abs(C).max(-1) == 0
    for car in ('lmean', 'floor'):
        S = carriers(X)[car]; bw = bands(S); bc = bands(C); s = '%-5s %-5s' % (nm, car)
        for j in range(len(bw)):
            best = None
            for th in (0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12):
                sf = soft(bw[j], th); a = float((sf * bc[j] * W).sum() / max((sf * sf * W).sum(), 1e-9)); a = min(max(a, 0.0), 1.0)
                for aa in (a, 0.75 * a, 0.5 * a):
                    e = np.abs(aa * sf - bc[j]) * W; tot = float(e.sum() / W.sum())
                    if best is None or tot < best[0]: best = (tot, th, aa, float(e[~nul].sum() / W[~nul].sum()), float(e[nul].sum() / W[nul].sum()))
            e0 = float((np.abs(bc[j]) * W)[~nul].sum() / W[~nul].sum()); s += ' | th %4.1f g %.2f: %.2f/%.2f c%.2f' % (best[1], best[2], best[3], e0, best[4])
        print(s, flush=True)
