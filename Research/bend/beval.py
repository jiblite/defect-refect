# The bend meter, blind: on the scoreboard's distortion and saturation cases and on those of sets2 (their truth is known):
# how sure it is, how far the squeeze it reads is off the true one, which name the group would get, and what taking out and
# putting on score with the read answer. Then how sure it is on every other scoreboard sound and on the clean corpus
# (kinds only, never names).
#   python3 beval.py meter.pt|.bin [K ...]          FAST=1 in the environment: no scores, only the reading
import os, pickle
from common import *
import bfeat, bengine, bmeter
def align(x, y):
    """By how many samples the sound after the box lies behind the sound before it (within four each way)."""
    cc = ss.correlate(y[:, 0] + y[:, 1], x[:, 0] + x[:, 1], 'full', 'fft'); n = len(x); k = np.arange(-4, 5) + n - 1; return int(k[np.argmax(np.abs(cc[k]))]) - (n - 1)
ROOTS = [(SETS, 'board'), ('/tmp/claude-0/v2/four/sets2/', 'new')]; CACHE = 'data/beval_X.pkl'
def feats(tag, y, cache):
    if tag not in cache: cache[tag] = bfeat.features(y)
    return cache[tag]
def answer(c, cache, tag):
    if tag not in cache:
        lag = align(c['dry'], c['wet']); pol = 1.0 if np.sum(c['dry'] * np.roll(c['wet'], -lag, 0)) >= 0 else -1.0; cache[tag] = np.clip(bfeat.answer(pol * np.roll(c['dry'], lag, 0), c['wet']), 0.0, 40.0)
    return cache[tag]
if __name__ == '__main__':
    meter = sys.argv[1]; Ks = [float(k) for k in sys.argv[2:]] or [bengine.SPREAD]; M = bmeter.model(meter); fast = bool(os.environ.get('FAST'))
    cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}; n0 = len(cache); rows = []; top = lambda S: max(bengine.tidy(S[0], bengine.SPREAD)[-1], bengine.tidy(S[1], bengine.SPREAD)[-1])
    for root, tag in ROOTS:
        for fam in ('distortion', 'saturation'):
            for d in sorted(glob.glob(root + fam + '/*/')):
                A = truth.load(d, 'A'); B = truth.load(d, 'B'); res = {}
                for w, c in (('A', A), ('B', B)):
                    key = '%s|%s|%s|%s' % (tag, fam, c['name'], w); T = answer(c, cache, key + '|T'); X, vp = feats(key, c['wet'], cache); S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1]))
                    res[w] = dict(S=S, vp=vp, logit=logit, T=T, off=float(np.sqrt(np.mean((np.clip(S, 0, 40) - T) ** 2))), top=top(S), ttop=top(T))
                a = res['A']; sc = ''; vals = []
                if not fast:
                    for K in Ks:
                        o = score.removed(A['dry'], A['wet'], bengine.own(A['wet'], a['S'], a['vp'], 0.0, K))[0]; p = score.removed(B['wet'], B['dry'], bengine.onto(B['dry'], a['S'], a['vp'], 1.0, K))[0]; vals += [o, p]; sc += ' | K %2.0f: out %5.0f onto %5.0f' % (K, o, p)
                rows.append((tag, fam, a['logit'], res['B']['logit'], a['top'], a['ttop'], a['off'], res['B']['off'], res['B']['top'], res['B']['ttop']) + tuple(vals))
                print('%-5s %-19s sure %5.1f %5.1f | top read %4.1f true %4.1f (B: %4.1f, %4.1f) | off by %4.1f %4.1f dB rms%s | read + %s | true + %s' % (tag, A['name'], a['logit'], res['B']['logit'], a['top'], a['ttop'], res['B']['top'], res['B']['ttop'],
                      a['off'], res['B']['off'], sc, ' '.join('%.0f' % v for v in np.clip(a['S'][0], 0, 40)), ' '.join('%.0f' % v for v in a['T'][0])), flush=True)
    R = np.array([r[2:] for r in rows], float); sure = np.concatenate([R[:, 0], R[:, 1]]); tops = np.concatenate([R[:, 2], R[:, 6]]); ttops = np.concatenate([R[:, 3], R[:, 7]])
    RULES = [('at the default', lambda l, t: (l >= bengine.SURE_AT) & (t >= bengine.SURE_TOP)), ('at all', lambda l, t: (l >= bengine.LOW_AT) & (t >= bengine.TOP_AT))]
    print('sounds shown %s of %d | squeeze off by %.1f dB rms on average (median %.1f) | the name (hard from %.0f dB): right on %d of %d' % (', '.join('%s: %d' % (n, f(sure, tops).sum()) for n, f in RULES), len(sure),
          R[:, 4:6].mean(), np.median(R[:, 4:6]), bengine.HARD_AT, ((tops >= bengine.HARD_AT) == (ttops >= bengine.HARD_AT)).sum(), len(tops)))
    for n, f in RULES:
        for lab, m in (('distortion cases', np.array([r[1] == 'distortion' for r in rows] * 2)), ('saturation cases, true top 6 dB and more', np.array([r[1] == 'saturation' for r in rows] * 2) & (ttops >= 6)), ('saturation cases, true top under 6 dB', np.array([r[1] == 'saturation' for r in rows] * 2) & (ttops < 6))):
            print('   shown %s: %d of %d sounds of the %s' % (n, (f(sure, tops) & m).sum(), m.sum(), lab))
    if not fast:
        for i, K in enumerate(Ks):
            o = R[:, 8 + 2 * i]; p = R[:, 9 + 2 * i]
            for n, rule in RULES:
                f = rule(R[:, 0], R[:, 2]); hard = R[:, 2] >= bengine.HARD_AT
                print('K %2.0f, shown %s: %d of %d cases | out mean %.0f, over the shown %.0f (median %.0f), worse than nothing %d | onto mean %.0f, over the shown %.0f (median %.0f), worse %d   (a case not shown counts 0 in the means)' % (K, n, f.sum(), len(f),
                      np.where(f, o, 0).mean(), o[f].mean(), np.median(o[f]), (np.where(f, o, 0) < 0).sum(), np.where(f, p, 0).mean(), p[f].mean(), np.median(p[f]), (np.where(f, p, 0) < 0).sum()))
                for lab, m in (('DISTORTION', f & hard), ('SATURATION', f & ~hard)):
                    if m.any(): print('      shown as %s: %d | out over them: mean %.0f median %.0f, worse than nothing %d | onto: mean %.0f median %.0f, worse %d' % (lab, m.sum(), o[m].mean(), np.median(o[m]), (o[m] < 0).sum(), p[m].mean(), np.median(p[m]), (p[m] < 0).sum()))
    # every other scoreboard sound, and the clean corpus
    fams = {}
    for fam in sorted(os.listdir(SETS)):
        if fam in ('distortion', 'saturation'): continue
        for d in sorted(glob.glob(SETS + fam + '/*/')):
            for w in 'AB':
                key = 'board|%s|%s|%s' % (fam, os.path.basename(d[:-1]), w)
                if key not in cache: cache[key] = bfeat.features(truth.load(d, w)['wet'])
                X, vp = cache[key]; S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); fams.setdefault(fam, []).append((logit, top(S)))
    clean = []
    for p in sorted(glob.glob(CORPUS + 'dry/*')):
        key = 'clean|' + os.path.basename(p)
        if key not in cache: cache[key] = bfeat.features(sf.read(p, dtype='float64', always_2d=True)[0])
        X, vp = cache[key]; S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); clean.append((logit, top(S), kind(os.path.basename(p))))
    if len(cache) != n0: pickle.dump(cache, open(CACHE, 'wb'))
    for n, rule in RULES:
        hit = lambda v: sum(1 for q in v if rule(np.float64(q[0]), np.float64(q[1])))
        print('shown %s: clean corpus %d of %d (%s) |' % (n, hit(clean), len(clean), ' '.join(sorted(q[2] for q in clean if rule(np.float64(q[0]), np.float64(q[1]))))), ' '.join('%s %d/%d' % (f, hit(v), len(v)) for f, v in sorted(fams.items())),
              '| all other effects together: %d of %d' % (sum(hit(v) for v in fams.values()), sum(len(v) for v in fams.values())))
