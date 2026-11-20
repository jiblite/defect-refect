# Which single number sorts real bends from the rest best: the verdict's logit, the squeeze read at the peak, or the two
# together? For each, the number of real bends kept when the limit is set so that 0, 2, 4, 8 of the 198 clean sounds pass.
import os, pickle, sys
from common import *
import bfeat, bengine, bmeter
M = bmeter.model(sys.argv[1]); cache = pickle.load(open('data/beval_X.pkl', 'rb')); top = lambda S: max(bengine.tidy(S[0], bengine.SPREAD)[-1], bengine.tidy(S[1], bengine.SPREAD)[-1])
G = {'dist': [], 'sat6': [], 'sat3': [], 'sat0': [], 'clean': [], 'other': []}
for key, val in cache.items():
    if key.endswith('|T') or not isinstance(val, tuple): continue
    X, vp = val; S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); t = top(S); p = key.split('|')
    if p[0] == 'clean': G['clean'].append((logit, t))
    elif p[1] == 'distortion': G['dist'].append((logit, t))
    elif p[1] == 'saturation': tt = top(cache[key + '|T']); G['sat6' if tt >= 6 else 'sat3' if tt >= 3 else 'sat0'].append((logit, t))
    else: G['other'].append((logit, t))
G = {k: np.array(v) for k, v in G.items()}
rules = [('the logit alone (top at least 3 dB)', lambda v: np.where(v[:, 1] >= 3, v[:, 0], -99)), ('the squeeze at the peak alone', lambda v: v[:, 1]),
         ('logit + 0.25 x squeeze', lambda v: np.where(v[:, 1] >= 3, v[:, 0] + 0.25 * v[:, 1], -99)), ('logit + 0.5 x squeeze', lambda v: np.where(v[:, 1] >= 3, v[:, 0] + 0.5 * v[:, 1], -99)), ('logit + 1 x squeeze', lambda v: np.where(v[:, 1] >= 3, v[:, 0] + 1.0 * v[:, 1], -99)),
         ('the larger of logit and squeeze - 6', lambda v: np.where(v[:, 1] >= 3, np.maximum(v[:, 0], v[:, 1] - 6.0), -99))]
for name, f in rules:
    c = np.sort(f(G['clean']))[::-1]; row = '%-38s' % name
    for k in (0, 2, 4, 8):
        thr = c[k] + 1e-9 if k < len(c) else -1e9          # just above the (k+1)-th highest clean sound: k clean sounds pass
        row += ' | %d clean: dist %2d/48 sat6+ %2d/25 sat3-6 %d/9 sat<3 %2d/14 other %2d/608 (from %.1f)' % (k, (f(G['dist']) >= thr).sum(), (f(G['sat6']) >= thr).sum(), (f(G['sat3']) >= thr).sum(), (f(G['sat0']) >= thr).sum(), (f(G['other']) >= thr).sum(), thr)
    print(row)
