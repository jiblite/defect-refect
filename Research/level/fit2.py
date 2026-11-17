# The rule "is a shared level movement (tremolo, gate) there": fitted on what find() measures about its candidates on clean 6 s train
# clips (never a movement: label 0) and on the same clips with one synthetic movement put on (the candidate with the true period: 1).
# Scores for the cut are out-of-fold (clips kept together).      python3 fit2.py [--write]   -> rule.json
import json, sys, glob, numpy as np, collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
rows = [json.loads(l) for f in sorted(glob.glob('/tmp/claude-0/v2/level/data/train_*.jsonl')) for l in open(f)]
FS = 1000.0
def match(P, rate, tol=0.012):
    """1: the true period; 2..4: that many true periods; -1: related some other way (half, 2/3, ...); 0: nothing to do with it."""
    r = (FS / P) / rate
    if abs(r - 1) < tol: return 1
    for m in (2, 3, 4):
        if abs(r * m - 1) < tol: return m
    for a in (1, 2, 3, 4):
        for b in (1, 2, 3, 4, 5, 6):
            if abs(r * b / a - 1) < 0.02: return -1
    return 0
def feats(m):
    sp = m['slope_hi'] - m['slope_lo']
    return [min(m['depth'], 40.0), np.log(max(m['depth'], 0.1)), m['corr_lo'], m['corr_med'], m['corr_min'], min(sp, 3.0), min(abs(m['slope_med'] - 1.0), 2.0), min(max(m['slope_min'], -1.0), 2.0), min(max(m['slope_max'], 0.0), 4.0),
            min(m['cyc_lo'], 2.0), min(abs(m['cyc_med'] - 1.0), 2.0), min(m['cyc_hi'] - m['cyc_lo'], 3.0), m['halves'], m['r2'], m['r2c'], m['share'], m['excess'], np.log(max(m['cyc_n'], 1)), m['steep'], m['fund'], m['silent'], m['skipfrac']]
NAMES = ['depth', 'logdepth', 'corr_lo', 'corr_med', 'corr_min', 'slope_spread', 'slope_off', 'slope_min', 'slope_max', 'cyc_lo', 'cyc_off', 'cyc_spread', 'halves', 'r2', 'r2c', 'share', 'excess', 'logcyc', 'steep', 'fund', 'silent', 'skipfrac']
X, y, grp, tag = [], [], [], []
for r in rows:
    for m in r['clean']['tog']: X.append(feats(m)); y.append(0); grp.append(r['name']); tag.append(('clean', r['name'], m['P']))
    if r['par']['kind'] in ('trem', 'gate'):
        for m in r['wet']['tog']:
            mt = match(m['P'], r['par']['rate'])
            if mt == 1: X.append(feats(m)); y.append(1); grp.append(r['name']); tag.append(('wet', r['name'], m['P']))
            elif mt == 0: X.append(feats(m)); y.append(0); grp.append(r['name']); tag.append(('wetoff', r['name'], m['P']))
X = np.array(X, float); y = np.array(y); grp = np.array(grp); X[~np.isfinite(X)] = 0.0
print('%d clips, %d candidates: %d true, %d not' % (len(rows), len(y), y.sum(), (1 - y).sum()))
mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd
def oof(make, Zm):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Zm, y, grp):
        mdl = make().fit(Zm[tr], y[tr]); s[te] = mdl.decision_function(Zm[te])
    return s
def clip_view(s, label):
    """Per clip: the best score of a clean clip's candidates (a false report if over the cut); whether the best-scoring candidate of a
    clip with a movement is the true one and over the cut."""
    best_clean = collections.defaultdict(lambda: -1e9);
    for (t, n, P), v in zip(tag, s):
        if t == 'clean': best_clean[n] = max(best_clean[n], v)
    bc = np.array(sorted(best_clean.values())); nclean = sum(1 for r in rows if True); bc = np.concatenate([np.full(nclean - len(bc), -1e9), bc])
    out = {}
    for far in (0.005, 0.01, 0.02, 0.05):
        cut = float(np.sort(bc)[::-1][int(np.floor(far * len(bc)))]) + 1e-9          # that many clean clips lie above
        got = collections.Counter(); tot = collections.Counter()
        best_true = collections.defaultdict(lambda: -1e9)
        for (t, n, P), v in zip(tag, s):
            if t == 'wet': best_true[n] = max(best_true[n], v)
        for r in rows:
            k = r['par']['kind']
            if k not in ('trem', 'gate'): continue
            d = 'deep' if r['par']['depth'] >= 0.6 else 'shallow'; tot[k] += 1; tot[k + ' ' + d] += 1
            if best_true[r['name']] >= cut: got[k] += 1; got[k + ' ' + d] += 1
        out[far] = (cut, {k: (got[k], tot[k]) for k in sorted(tot)})
    print(label);
    for far, (cut, g) in out.items(): print('   clean clips over the cut %.1f%% (cut %.2f): ' % (100 * far, cut) + '  '.join('%s %d/%d (%.0f%%)' % (k, a, b, 100 * a / max(b, 1)) for k, (a, b) in g.items()))
    return out, best_clean
sl = oof(lambda: LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'), Z); ol, bcl = clip_view(sl, 'logistic, all measures (out of fold):')
sg = oof(lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=0), Z); og, bcg = clip_view(sg, 'boosted trees (out of fold):')
mdl = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(Z, y); w = mdl.coef_[0] / sd; b = float(mdl.intercept_[0] - (mdl.coef_[0] * mu / sd).sum())
print('weights: ' + '  '.join('%s %+.2f' % (n, c) for n, c in sorted(zip(NAMES, mdl.coef_[0]), key=lambda q: -abs(q[1]))))
top = sorted(bcl.items(), key=lambda q: -q[1])[:14]; print('clean clips with the highest scores (logistic): ' + '  '.join('%s %.1f' % q for q in top))
top = sorted(bcg.items(), key=lambda q: -q[1])[:14]; print('clean clips with the highest scores (trees): ' + '  '.join('%s %.1f' % q for q in top))
if '--write' in sys.argv:
    far = float(sys.argv[sys.argv.index('--write') + 1]) if len(sys.argv) > sys.argv.index('--write') + 1 else 0.01
    json.dump(dict(names=NAMES, w=[float(v) for v in w], b=b, cut=ol[far][0]), open('/tmp/claude-0/v2/level/rule.json', 'w')); print('written: cut', ol[far][0])
