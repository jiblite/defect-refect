# The bend engine's way back, second look: a lower cap on how far values are spread apart (K), with the read squeeze in part
# and less a margin. Out scores on the outside cases.
#   python3 bback2.py meter[,meter...]
import os, pickle
from common import *
import bfeat, bengine, bmeter
ROOTS = [(SETS, 'board'), ('/tmp/claude-0/v2/four/sets2/', 'new')]; cache = pickle.load(open('data/beval_X.pkl', 'rb')); M = bmeter.model(sys.argv[1])
part = lambda S: np.maximum(0.75 * S - 1.0, 0)
VAR = [('as read, K 6', lambda S: S, 6.0), ('0.7, K 6', lambda S: 0.7 * S, 6.0), ('0.75 less 1, K 6', part, 6.0), ('less 1, K 2', lambda S: np.maximum(S - 1.0, 0), 2.0), ('0.7, K 2', lambda S: 0.7 * S, 2.0), ('0.75 less 1, K 2', part, 2.0), ('0.75 less 1, K 3', part, 3.0), ('true, K 2', None, 2.0)]
rows = []
for root, tag in ROOTS:
    for fam in ('distortion', 'saturation'):
        for d in sorted(glob.glob(root + fam + '/*/')):
            A = truth.load(d, 'A'); key = '%s|%s|%s|A' % (tag, fam, A['name']); X, vp = cache[key]; T = cache[key + '|T']; S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); S = np.clip(S, 0, 40)
            top = max(bengine.tidy(S[0], 6.0)[-1], bengine.tidy(S[1], 6.0)[-1]); r = [logit, top]
            for name, f, K in VAR: r.append(score.removed(A['dry'], A['wet'], bengine.own(A['wet'], T if f is None else f(S), vp, 0.0, K, plain=True))[0])
            rows.append(r); print('%-5s %-19s sure %5.1f top %4.1f | out: %s' % (tag, A['name'], logit, top, ' '.join('%6.0f' % v for v in r[2:])), flush=True)
R = np.array(rows)
for thr in (0.0, 1.5):
    f = (R[:, 0] >= thr) & (R[:, 1] >= bengine.TOP_AT); print('shown from %.1f: %d of %d cases' % (thr, f.sum(), len(f)))
    for i, (n, _, _) in enumerate(VAR):
        o = R[f, 2 + i]; hard = R[f, 1] >= bengine.HARD_AT
        print('   out, %-18s over the shown: mean %5.1f median %5.1f | worse than nothing on %2d, under -20 on %2d | DISTORTION (%d): mean %5.1f median %5.1f | SATURATION (%d): mean %5.1f median %5.1f' % (n + ':', o.mean(), np.median(o), (o < 0).sum(), (o < -20).sum(),
              hard.sum(), o[hard].mean(), np.median(o[hard]), (~hard).sum(), o[~hard].mean(), np.median(o[~hard])))
pickle.dump(R, open('data/bback2_R.pkl', 'wb'))
