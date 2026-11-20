# The fixed filter's limits. For one reader or several read together: what the app would show (the reader's verdict, the
# depth of the cut, the kind of filter) on the 32 outside cases with their scores, on the clean corpus and on the
# scoreboard's sounds of the other effects; then, for a grid of limits, what is shown and what it scores.
#   python3 fgrid.py make tag how net.pt [net.pt ...]     how: mean | least ; writes data/fgrid_<tag>.pkl (FAST=1: no scores)
#   python3 fgrid.py show tag [tag ...]
import os, pickle, collections
from common import *
import ffeat, fengine
ROOTS = ['/tmp/claude-0/v2/bench/sets/filter/', '/tmp/claude-0/v2/four/sets2/filter/']; CACHE = 'data/feval_E.pkl'
def skirt_on(L, at):
    """The skirt runs on: outward from the curve's highest point, once the curve has fallen `at` dB under the pass band it
    does not rise again. (Where a filter has cut everything the reader has nothing to read, and its answer drifts back up.)"""
    L = np.array(L, float); top = int(np.argmax(L))
    for bands in (range(top + 1, len(L)), range(top - 1, -1, -1)):
        m = L[top]
        for b in bands:
            m = min(m, L[b])
            if m <= -at: L[b] = m
    return L
def skirt_floor(L, E, at, near=6.0):
    """The same, but only in bands where the sound holds nothing (its level within `near` dB of the lowest the reader is
    given): there nothing can be read, and the curve keeps the lowest point it reached on the way out from the pass band."""
    X, ref = ffeat.curves6(np.asarray(E, np.float64)); silent = X[0] <= ffeat.FLOOR + near; L = np.array(L, float); top = int(np.argmax(L))
    for bands in (range(top + 1, len(L)), range(top - 1, -1, -1)):
        m = L[top]
        for b in bands:
            m = min(m, L[b])
            if silent[b] and m <= -at: L[b] = m
    return L
VARIANTS = [('as read', None), ('skirt on from 12 dB', 12.0), ('skirt on from 24 dB', 24.0)]
if os.environ.get('FLOORV'): VARIANTS = [('as read', None), ('nothing there: from 12 dB', -12.0), ('nothing there: from 6 dB', -6.0), ('nothing there: from 24 dB', -24.0)]
def shape(L, E, at): return L if at is None else skirt_on(L, at) if at > 0 else skirt_floor(L, E, -at)
def verdict(F, E):
    """As Filter.h's find(): the deepest cut where the clean sound would be heard, the kind (1 low-pass, 2 high-pass, 3 band-pass,
    0 none), and how far the nets lie apart on average where the sound is heard."""
    X, ref = ffeat.curves6(np.asarray(E, np.float64)); L = F['curve']; clean = X[0] - L; aud = clean > clean.max() - 50.0; depth = float(max(np.max(-L[aud]), 0.0)); top = int(np.argmax(L))
    kind = 0 if L[top] <= -3.0 else (1 if L[-1] <= -6.0 else 0) + (2 if L[0] <= -6.0 else 0)
    return depth, kind, float(F['apart'][X[0] > X[0].max() - 50.0].mean())
def make(tag, how, nets):
    cache = pickle.load(open(CACHE, 'rb')); fast = bool(os.environ.get('FAST')); cases_ = []; clean = []; other = []
    for root in ROOTS:
        for d in sorted(glob.glob(root + '*/')):
            A = truth.load(d, 'A'); B = truth.load(d, 'B'); t = ('new ' if 'sets2' in root else 'board ') + A['name']; r = dict(name=t)
            for w, c in (('A', A), ('B', B)):
                Ed = cache[t + w + 'dry']; Ew = cache[t + w + 'wet']; T = np.clip(10 * np.log10(Ew.mean(0) + 1e-30) - 10 * np.log10(Ed.mean(0) + 1e-30), -60, 24); Sx = 10 * np.log10(Ed.mean(0) + 1e-30); aud = Sx > Sx.max() - 50
                F0 = fengine.find(c['wet'], nets, 2, how, E=Ew.astype(np.float64), plain=True); r[w] = {}
                for vn, at in VARIANTS:
                    F = dict(F0, curve=shape(F0['curve'], Ew, at)); F['lift'] = fengine.lift_cap(Ew.astype(np.float64), F['curve']); depth, kd, apart = verdict(F, Ew)
                    r[w][vn] = dict(sure=F['sure'], depth=depth, kind=kd, apart=apart, off=float(np.sqrt(np.mean((T - np.clip(F['curve'], -60, 24))[aud] ** 2))), true=float(np.max(-T[aud])), out=np.nan, onto=np.nan)
                    if w == 'A' and not fast: r[w][vn]['out'] = score.removed(A['dry'], A['wet'], fengine.own(A['wet'], F, 0.0))[0]; r[w][vn]['onto'] = score.removed(B['wet'], B['dry'], fengine.onto(B['dry'], F, 1.0))[0]
            cases_.append(r); print('%-24s sure %5.1f true depth %4.1f |' % (t, r['A']['as read']['sure'], r['A']['as read']['true']), ' | '.join('%s: depth %4.1f kind %d out %4.0f onto %4.0f' % (vn[-10:], r['A'][vn]['depth'], r['A'][vn]['kind'], r['A'][vn]['out'], r['A'][vn]['onto']) for vn, _ in VARIANTS), flush=True)
    C = np.load('data/corpus_E.npz')
    for e, n in zip(C['E'], C['names']):
        F0 = fengine.find(None, nets, 2, how, E=e.astype(np.float64), plain=True); r = dict(kind_of=kind(str(n)))
        for vn, at in VARIANTS: F = dict(F0, curve=shape(F0['curve'], e, at)); depth, kd, apart = verdict(F, e); r[vn] = dict(sure=F['sure'], depth=depth, kind=kd, apart=apart)
        clean.append(r)
    Z = np.load('data/cases_E.npz')
    for k in Z.files:
        fam, name, w = k.split('|')
        if fam == 'filter': continue
        F0 = fengine.find(None, nets, 2, how, E=Z[k].astype(np.float64), plain=True); r = dict(fam=fam)
        for vn, at in VARIANTS: F = dict(F0, curve=shape(F0['curve'], Z[k], at)); depth, kd, apart = verdict(F, Z[k]); r[vn] = dict(sure=F['sure'], depth=depth, kind=kd, apart=apart)
        other.append(r)
    pickle.dump(dict(cases=cases_, clean=clean, other=other, how=how, nets=nets), open('data/fgrid_%s.pkl' % tag, 'wb'))
def show(tag, vn, aparts=(99.0,), deeps=(6.0, 12.0), ats=(0.0, 1.0, 2.0, 3.0, 4.0)):
    R = pickle.load(open('data/fgrid_%s.pkl' % tag, 'rb')); cs = R['cases']; print('== %s (%s of %d), %s' % (tag, R['how'], len(R['nets']), vn))
    A = np.array([(c['A'][vn]['sure'], c['A'][vn]['depth'], c['A'][vn]['kind'], c['A'][vn]['apart'], c['A'][vn]['out'], c['A'][vn]['onto']) for c in cs]); S = np.array([(c[w][vn]['sure'], c[w][vn]['depth'], c[w][vn]['kind'], c[w][vn]['apart']) for c in cs for w in 'AB'])
    cl = np.array([(c[vn]['sure'], c[vn]['depth'], c[vn]['kind'], c[vn]['apart']) for c in R['clean']]); ot = np.array([(c[vn]['sure'], c[vn]['depth'], c[vn]['kind'], c[vn]['apart']) for c in R['other']]); fam = np.array([c['fam'] for c in R['other']])
    print('curve off by %.1f dB rms on average (median %.1f) over the %d sounds' % (np.mean([c[w][vn]['off'] for c in cs for w in 'AB']), np.median([c[w][vn]['off'] for c in cs for w in 'AB']), 2 * len(cs)))
    for apart in aparts:
        for deep in deeps:
            for at in ats:
                ok = lambda X: (X[:, 0] >= at) & (X[:, 1] >= deep) & (X[:, 2] > 0) & (X[:, 3] <= apart); f = ok(A); g = ok(ot); o = A[f, 4]; p = A[f, 5]
                if not f.any(): continue
                print('nets apart up to %4.0f, cut from %2.0f dB, verdict from %1.0f: sounds %2d of 64, cases %2d of 32 | out mean %3.0f median %3.0f worse %d (under -20: %d) | onto mean %3.0f median %3.0f worse %d (under -20: %d) | clean %2d of %d | others %3d of %d: %s' % (
                    apart, deep, at, ok(S).sum(), f.sum(), o.mean(), np.median(o), (o < 0).sum(), (o < -20).sum(), p.mean(), np.median(p), (p < 0).sum(), (p < -20).sum(), ok(cl).sum(), len(cl), g.sum(), len(g),
                    ' '.join('%s %d' % (k[:6], v) for k, v in sorted(collections.Counter(fam[g]).items()))))
if __name__ == '__main__':
    if sys.argv[1] == 'make': make(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        for t in sys.argv[2:]:
            for vn, _ in VARIANTS: show(t, vn)
