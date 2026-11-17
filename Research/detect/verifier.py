# The models of the verification step (Source/Core/Verify.h): per effect gradient-boosted trees (the kind the app already
# reads: Models.h Boosted) that see what the app's two detectors compute, gathered over the windows (mean / max / min of
# the CNN outputs, the hand numbers and the deciders' scores), plus the whole-sound physical evidence (evidence.py).
# Trained on many more kinds of clean sound than the detectors' own deciders saw, and on outside effects as well as the app's own.
#   python3 verifier.py oof  VARIANT     out-of-fold scores on the training sounds (folds by source; outside effects scored by
#                                        models that never saw that implementation)  -> data/oof_VARIANT.npz
#   python3 verifier.py final VARIANT    thresholds from the out-of-fold scores, models on all training sounds, scores for the
#                                        held-out test sounds -> model/verifier_VARIANT.pkl, data/test_VARIANT.npz
#   VARIANT: v2 (what the app has: everything the detection computes + the 55 numbers of evidence.py) | all (v2 + the 29 further
#            numbers of the first feature run) | app (no evidence) | ev (evidence only) | emb (all + the CNNs' embeddings)
import sys, os, glob, pickle, json, time
import paths
import numpy as np
D = paths.WORK; FE = os.environ.get('FEATS', D + 'data/feats/')
F12 = ['flanger', 'chorus', 'phaser', 'haas', 'width', 'delay', 'reverb', 'distortion', 'bitcrush', 'filtersweep', 'tremolo', 'autopan']
NEW = ['saturation', 'exciter', 'ringmod', 'filter', 'stutter', 'gatedrhythm']
F27 = F12 + ['eq', 'compressor', 'limiter', 'gate', 'transient', 'saturation', 'exciter', 'ringmod', 'freqshift', 'pitchshift', 'wowflutter', 'filter', 'reverse', 'stutter', 'gatedrhythm']
FIN = F12 + NEW; FAMS = [f for f in F12 if f != 'reverb'] + NEW
HAND = ['echo_sc', 'echo_ms', 'side_ratio', 'corr', 'haas_z', 'haas_ms', 'pan_r2', 'pan_span', 'clip_frac', 'log_unique', 'hold', 'crest', 'kurt', 'trem_r2', 'trem_depth']
EVN = json.load(open(D + 'data/ev_names.json')) if os.path.exists(D + 'data/ev_names.json') else None
def cut_for(f, cut=0.65): return min(0.99, max(cut + 0.15, 0.8 if cut == 0.65 else 0.0)) if f in NEW else cut
def app_shown(r, cut=0.65): return [f for f in FAMS if r['final'][FIN.index(f)] > cut_for(f, cut)]

def feat_names():
    n = []
    for key, cols in (('lg12', F12), ('lg27', F27), ('hd', HAND), ('p12', F12), ('p27', F27)):
        for st in ('mean', 'max', 'min'): n += ['%s_%s_%s' % (key, c, st) for c in cols]
    return n + ['fin_' + c for c in FIN] + ['wf_sc', 'wf_ms'] + list(EVN)
def vec(r):
    out = []
    for key in ('logit12', 'logit27', 'hand', 'p12', 'p27'):
        a = np.asarray(r[key], np.float32)
        if a.shape[0] == 0: a = np.zeros((1, a.shape[1]), np.float32)
        out += [a.mean(0), a.max(0), a.min(0)]
    out += [r['final'], r['echo'], r['ev']]
    return np.concatenate([np.asarray(v, np.float32).ravel() for v in out])
def emb(r): return np.concatenate([np.asarray(r[k], np.float32) for k in ('emb12m', 'emb27m')]) if 'emb12m' in r else np.zeros(768, np.float32)
DROP = ('sm_', 'ro_', 'cb_', 'sw_', 'ec_')      # evidence left out of the app's verifier (variant v2): the old sweep check (slow), spectrum-shape and roll-off movement, onset echoes
def columns(variant, names):
    if variant in ('all', 'emb'): return np.arange(len(names))
    if variant == 'v2': return np.flatnonzero([not n.startswith(DROP) for n in names])
    isev = np.array([n in EVN for n in names])
    return np.flatnonzero(isev) if variant == 'ev' else np.flatnonzero(~isev)

def load(prefixes):
    rows = []
    for p in prefixes:
        for f in sorted(glob.glob(FE + p + '*.pkl')): rows += pickle.load(open(f, 'rb'))
    return rows
def train_rows():
    """Training sounds, without the recordings that have another excerpt among the test clips (and without mixes made with one)."""
    import gen_feats as G
    sib = { G.group_of(k) for k in G.SIBLING }; rows = load(['src_', 'casesdev_', 'pos2_'])
    return [r for r in rows if r['group'] not in sib and G.group_of(r.get('with') or r['src']) not in sib]
def test_rows(): return load(['t_'])

def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, l2_regularization=1.0, max_bins=64, early_stopping=False, random_state=0)
def fit(X, y):
    m = model(); m.fit(X, y); return m
def score(m, X): return m.predict_proba(X)[:, 1]

def matrices(rows, variant):
    names = feat_names(); X = np.stack([vec(r) for r in rows]); cols = columns(variant, names); X = X[:, cols]; names = [names[i] for i in cols]
    if variant == 'emb': X = np.hstack([X, np.stack([emb(r) for r in rows])]); names += ['e12_%d' % i for i in range(384)] + ['e27_%d' % i for i in range(384)]
    Y = np.array([[f in r['labels'] for f in FAMS] for r in rows]); return X, Y, names
def outside_impls(rows):
    out = {}
    for r in rows:
        for f, i in zip(r['labels'], r['impl']):
            if i != 'own': out.setdefault(f, set()).add(i)
    return { f: sorted(v) for f, v in out.items() }
def has_impl(r, f, name): return any(a == f and b == name for a, b in zip(r['labels'], r['impl']))

def oof(variant, fams=None):
    from cpuslot import slot
    rows = train_rows(); X, Y, names = matrices(rows, variant); fold = np.array([r['fold'] for r in rows]); imp = outside_impls(rows); S = np.zeros((len(rows), len(FAMS)), np.float32); t0 = time.time()
    for k, f in enumerate(FAMS):
        if fams and f not in fams: continue
        with slot():
            for fo in range(5):
                te = fold == fo; tr = ~te; S[te, k] = score(fit(X[tr], Y[tr, k]), X[te])
                for name in imp.get(f, []):                    # sounds made with this outside effect: scored by a model that has not seen it
                    mine = np.array([has_impl(r, f, name) for r in rows]); sel = te & mine
                    if sel.any(): S[sel, k] = score(fit(X[tr & ~mine], Y[tr & ~mine, k]), X[sel])
        print('%s: out-of-fold done, %.0f s' % (f, time.time() - t0), flush=True)
    np.savez(D + 'data/oof_%s%s.npz' % (variant, ('_' + '_'.join(fams)) if fams else ''), S=S, ids=np.array([r['id'] for r in rows])); return S

def roc(sp, sn):
    sN = np.sort(sn)[::-1]; return [float(np.mean(sp > sN[k])) if k < len(sN) else 1.0 for k in range(len(sN) + 1)]
def allocate(curves, budget, weights=None):
    """Per effect how many clean sounds may lie above its cut, so that the found shares add up highest within a total budget."""
    best = np.zeros(budget + 1); choice = []
    for c in curves:
        c = np.array(c[:budget + 1]); nb = np.zeros(budget + 1); ch = np.zeros(budget + 1, int)
        for b in range(budget + 1):
            v = best[b::-1][:len(c[:b + 1])] + c[:b + 1]; k = int(np.argmax(v)); nb[b] = v[k]; ch[b] = k
        best = nb; choice.append(ch)
    ks = []; b = budget
    for ch in choice[::-1]: ks.append(int(ch[b])); b -= ks[-1]
    return ks[::-1]
def thresholds(S, Y, clean, budget_per_sound=0.07, floor=0.5):
    """Cuts from out-of-fold scores: the k-th highest clean score per effect, k from the budget split that finds most."""
    n = int(clean.sum()); B = int(budget_per_sound * n); curves = [roc(S[Y[:, k], k], S[clean, k]) for k in range(len(FAMS))]; ks = allocate(curves, B); T = []
    for k, kk in enumerate(ks):
        sN = np.sort(S[clean, k])[::-1]; hi = sN[kk] if kk < len(sN) else 0.0; lo_ = sN[kk + 1] if kk + 1 < len(sN) else 0.0
        T.append(float(max(floor, hi + 1e-6)))
    return np.array(T), ks

def final(variant, budget=0.07, floor=0.5):
    from cpuslot import slot
    rows = train_rows(); X, Y, names = matrices(rows, variant); clean = np.array([len(r['labels']) == 0 for r in rows])
    S = np.load(D + 'data/oof_%s.npz' % variant)['S']; assert len(S) == len(rows); T, ks = thresholds(S, Y, clean, budget, floor)
    te = test_rows(); Xt, Yt, _ = matrices(te, variant); St = np.zeros((len(te), len(FAMS)), np.float32); models = []
    for k, f in enumerate(FAMS):
        with slot(): m = fit(X, Y[:, k]); models.append(m); St[:, k] = score(m, Xt)
        print(f, 'cut %.3f (%d clean out-of-fold sounds above)' % (T[k], ks[k]), flush=True)
    paths.make_work()
    pickle.dump(dict(variant=variant, fams=FAMS, names=names, models=models, cuts=T, budget=budget, floor=floor), open(D + 'model/verifier_%s.pkl' % variant, 'wb'))
    np.savez(D + 'data/test_%s.npz' % variant, S=St, ids=np.array([r['id'] for r in te]), T=T)
if __name__ == '__main__':
    if EVN is None: print('data/ev_names.json missing'); sys.exit(1)
    if sys.argv[1] == 'oof': oof(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == 'final': final(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.07, float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
