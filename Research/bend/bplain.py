# How far is an outside box one curve? Per case (its first sound): the part of the box's output that one rising curve of
# its input, fitted through all pairs of values, leaves unexplained, in dB under the output; beside it what a plain gain
# leaves. A box that is a curve and nothing else leaves nothing (-300); one that filters before or behind its curve, or
# mixes delayed sound in, leaves a part that no curve can hold.
#   python3 bplain.py
import os, sys, re, numpy as np, collections
from common import *
from sklearn.isotonic import IsotonicRegression
ROOTS = [('/tmp/claude-0/v2/bench/sets/', 'board'), ('/tmp/claude-0/v2/four/sets2/', 'new')]
T = collections.defaultdict(list)
for root, tag in ROOTS:
    for fam in ('distortion', 'saturation'):
        for c in sorted(os.listdir(os.path.join(root, fam))):
            d = os.path.join(root, fam, c) + '/'; A = truth.load(d, 'A'); u = np.asarray(A['dry'], float); y = np.asarray(A['wet'], float)
            if u.ndim == 1: u = u[:, None]
            if y.ndim == 1: y = y[:, None]
            n = min(len(u), len(y)); u = u[:n, 0]; y = y[:n, 0]
            # the lag, within eight samples each way, at which the two fit best
            best = (1e9, 0)
            for lag in range(-8, 9):
                a = u[max(0, -lag): n - max(0, lag)]; b = y[max(0, lag): n - max(0, -lag)]
                cc = -abs(np.dot(a, b))
                if cc < best[0]: best = (cc, lag)
            lag = best[1]; a = u[max(0, -lag): n - max(0, lag)]; b = y[max(0, lag): n - max(0, -lag)]
            sub = slice(0, len(a), max(1, len(a) // 200000))
            iso = IsotonicRegression(increasing='auto', out_of_bounds='clip').fit(a[sub], b[sub]); f = iso.predict(a)
            r1 = 10 * np.log10(np.sum((b - f) ** 2) / np.sum(b ** 2) + 1e-30)
            lin = np.dot(a, b) / np.dot(a, a); r0 = 10 * np.log10(np.sum((b - lin * a) ** 2) / np.sum(b ** 2) + 1e-30)
            T[re.sub(r'_\d+$', '', c)].append((r0, r1, lag))
for t, v in sorted(T.items()):
    print('%-16s cases %d | left by a plain gain: %s dB | left by one rising curve: %s dB | lag %s' % (t, len(v), ' '.join('%4.0f' % x[0] for x in v), ' '.join('%4.0f' % x[1] for x in v), ' '.join(str(x[2]) for x in v)))
