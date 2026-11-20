# The bend engine's way back with a reading that is off by a dB or two: what taking out scores on the outside cases when the
# read squeeze is used as it is, less a margin, or in part; and with the true answer. Also onto a second sound.
#   python3 bback.py meter[,meter...]
import os, pickle
from common import *
import bfeat, bengine, bmeter
ROOTS = [(SETS, 'board'), ('/tmp/claude-0/v2/four/sets2/', 'new')]; cache = pickle.load(open('data/beval_X.pkl', 'rb')); M = bmeter.model(sys.argv[1]); K = bengine.SPREAD
VAR = [('as read', lambda S: S), ('less 1 dB', lambda S: np.maximum(S - 1.0, 0)), ('less 2 dB', lambda S: np.maximum(S - 2.0, 0)), ('0.7 of it', lambda S: 0.7 * S), ('half of it', lambda S: 0.5 * S)]
rows = []
for root, tag in ROOTS:
    for fam in ('distortion', 'saturation'):
        for d in sorted(glob.glob(root + fam + '/*/')):
            A = truth.load(d, 'A'); B = truth.load(d, 'B'); key = '%s|%s|%s|A' % (tag, fam, A['name']); X, vp = cache[key]; T = cache[key + '|T']; S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); S = np.clip(S, 0, 40)
            top = max(bengine.tidy(S[0], K)[-1], bengine.tidy(S[1], K)[-1]); r = [logit, top]
            for name, f in VAR + [('the true answer', lambda S: T)]: r.append(score.removed(A['dry'], A['wet'], bengine.own(A['wet'], f(S), vp, 0.0, K, plain=True))[0])
            r.append(score.removed(B['wet'], B['dry'], bengine.onto(B['dry'], S, vp, 1.0, K))[0]); r.append(score.removed(B['wet'], B['dry'], bengine.onto(B['dry'], T, vp, 1.0, K))[0]); rows.append(r)
            print('%-5s %-19s sure %5.1f top %4.1f | out: %s | onto: read %5.0f, true answer %5.0f' % (tag, A['name'], logit, top, ' '.join('%6.0f' % v for v in r[2:8]), r[8], r[9]), flush=True)
R = np.array(rows); names = [n for n, _ in VAR] + ['the true answer']
for thr in (0.0, 1.5, 3.0):
    f = (R[:, 0] >= thr) & (R[:, 1] >= bengine.TOP_AT); print('shown from %.1f: %d of %d cases (a case not shown counts 0)' % (thr, f.sum(), len(f)))
    for i, n in enumerate(names):
        o = np.where(f, R[:, 2 + i], 0); print('   out, %-16s mean %5.1f median %5.1f | worse than nothing on %2d, under -20 on %2d | over the shown: mean %5.1f median %5.1f' % (n + ':', o.mean(), np.median(o), (o < 0).sum(), (o < -20).sum(), R[f, 2 + i].mean(), np.median(R[f, 2 + i])))
    for i, n in ((8, 'read'), (9, 'the true answer')):
        o = np.where(f, R[:, i], 0); print('   onto, %-15s mean %5.1f median %5.1f | worse than nothing on %2d | over the shown: mean %5.1f median %5.1f' % (n + ':', o.mean(), np.median(o), (o < 0).sum(), R[f, i].mean(), np.median(R[f, i])))
pickle.dump(R, open('data/bback_R.pkl', 'wb'))
