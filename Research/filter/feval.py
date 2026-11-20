# The fixed-filter reader on cases with a known truth (the scoreboard's 8 and the 24 of sets2): how sure, how far the read
# curve is off, and what taking out and putting on score.
#   python3 feval.py rounds how net.pt [net.pt ...]         how: mean | least       (FAST=1 in the environment: no scores, only the reading)
import os, pickle
from common import *
import ffeat, fengine
ROOTS = ['/tmp/claude-0/v2/bench/sets/filter/', '/tmp/claude-0/v2/four/sets2/filter/']; CACHE = 'data/feval_E.pkl'
def energies_of(tag, y, cache):
    if tag not in cache: cache[tag] = ffeat.energies(y).astype(np.float32)
    return cache[tag]
if __name__ == '__main__':
    fengine.SKIRT = not os.environ.get('PLAIN'); rounds = int(sys.argv[1]); how = sys.argv[2]; nets = sys.argv[3:]; cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}; n0 = len(cache); fast = bool(os.environ.get('FAST')); rows = []
    for root in ROOTS:
        for d in sorted(glob.glob(root + '*/')):
            A = truth.load(d, 'A'); B = truth.load(d, 'B'); tag = ('new ' if 'sets2' in root else 'board ') + A['name']; res = {}
            for w, c in (('A', A), ('B', B)):
                Ed = energies_of(tag + w + 'dry', c['dry'], cache); Ew = energies_of(tag + w + 'wet', c['wet'], cache); T = np.clip(10 * np.log10(Ew.mean(0) + 1e-30) - 10 * np.log10(Ed.mean(0) + 1e-30), -60, 24)
                Sx = 10 * np.log10(Ed.mean(0) + 1e-30); F = fengine.find(c['wet'], nets, rounds, how, E=Ew.astype(np.float64)); aud = Sx > Sx.max() - 50; L = np.clip(F['curve'], -60, 24)
                res[w] = dict(F=F, sure=F['sure'], off=float(np.sqrt(np.mean((T - L)[aud] ** 2))), big=float(np.abs(T[aud]).max()), over=float(np.maximum(T - L, 0)[aud].max()), apart=float(F['apart'][aud].mean()))
            out = onto = np.nan
            if not fast: out = score.removed(A['dry'], A['wet'], fengine.own(A['wet'], res['A']['F'], 0.0))[0]; onto = score.removed(B['wet'], B['dry'], fengine.onto(B['dry'], res['A']['F'], 1.0))[0]
            rows.append((tag, res['A']['sure'], res['B']['sure'], res['A']['off'], res['B']['off'], out, onto, res['A']['over'], res['A']['apart']))
            print('%-24s sure %5.1f %5.1f | curve off by %4.1f %4.1f dB rms | out %4.0f onto %4.0f | cuts read %4.1f dB too deep at the most, nets apart %.1f dB' % rows[-1], flush=True)
    if len(cache) != n0: pickle.dump(cache, open(CACHE, 'wb'))
    R = np.array([r[1:] for r in rows], float); sure = np.concatenate([R[:, 0], R[:, 1]])
    print('sounds sure above 0 / 1 / 2 / 3 / 4: %s of %d | curve off %.1f dB rms on average (median %.1f)' % (' / '.join(str(int((sure > t).sum())) for t in (0, 1, 2, 3, 4)), len(sure), R[:, 2:4].mean(), np.median(R[:, 2:4])))
    if not fast:
        for thr in (0.0, 2.0, 3.0):
            f = R[:, 0] > thr; print('shown from %.0f: %d of %d cases | out mean %.0f median %.0f, worse than nothing %d | onto mean %.0f median %.0f, worse %d   (a case not shown counts 0)' % (thr, f.sum(), len(f),
                  np.where(f, R[:, 4], 0).mean(), np.median(np.where(f, R[:, 4], 0)), (np.where(f, R[:, 4], 0) < 0).sum(), np.where(f, R[:, 5], 0).mean(), np.median(np.where(f, R[:, 5], 0)), (np.where(f, R[:, 5], 0) < 0).sum()))
