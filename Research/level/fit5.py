# The rule "left and right move against each other (auto pan)": fitted on data/pan_train.jsonl (clean train clips, and the same with a
# synthetic pan put on).     python3 fit5.py [--write FAR]
import json, sys, numpy as np, collections, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level
rows = [json.loads(l) for l in open('/tmp/claude-0/v2/level/data/pan_train.jsonl')]; FS = 1000.0
def right(m, par):
    r = (FS / m['P']) / par['rate']; return bool(abs(r - 1) < 0.03 or abs(r * 2 - 1) < 0.03 or (m['slow'] and 0.6 < r < 1.6))
gate = lambda m: m is not None and np.isfinite(level.score_pan(m))
X, y, grp, tag = [], [], [], []
for r in rows:
    if gate(r['clean']): X.append(level.pan_features(r['clean'])); y.append(0); grp.append(r['name']); tag.append(('clean', r['name'], r['clean']))
    if gate(r['wet']) and right(r['wet'], r['par']): X.append(level.pan_features(r['wet'])); y.append(1); grp.append(r['name']); tag.append(('wet', r['name'], r['wet']))
X = np.array(X, float); X[~np.isfinite(X)] = 0.0; y = np.array(y); grp = np.array(grp); mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd
print('%d clips: clean with a left-right swing of 3 dB or more: %d; pans measured with the right period: %d of %d' % (len(rows), (y == 0).sum(), (y == 1).sum(), len(rows)))
s = np.zeros(len(y))
for tr, te in GroupKFold(5).split(Z, y, grp): s[te] = LogisticRegression(C=0.3, max_iter=5000, class_weight='balanced').fit(Z[tr], y[tr]).decision_function(Z[te])
bc = collections.defaultdict(lambda: -1e9); bt = collections.defaultdict(lambda: -1e9)
for (t, n, m), v in zip(tag, s):
    if t == 'clean': bc[n] = v
    else: bt[n] = v
a = np.sort(np.array([bc[r['name']] for r in rows]))[::-1]; cuts = {}
for far in (0.002, 0.005, 0.01, 0.02):
    cut = float(a[int(np.floor(far * len(a)))]) + 1e-9; cuts[far] = cut; got = collections.Counter(); tot = collections.Counter()
    for r in rows:
        p = r['par']; ks = ['all', 'law ' + p['law'], 'slow (under 0.4 Hz)' if p['rate'] < 0.4 else 'fast', 'deep' if p['depth'] >= 0.75 else 'shallow']
        for k in ks: tot[k] += 1; got[k] += bt[r['name']] >= cut
    print('clean clips reported %.1f%% (cut %.2f): ' % (100 * far, cut) + '  '.join('%s %d/%d (%.0f%%)' % (k, got[k], tot[k], 100 * got[k] / max(tot[k], 1)) for k in sorted(tot)))
mdl = LogisticRegression(C=0.3, max_iter=5000, class_weight='balanced').fit(Z, y); w = mdl.coef_[0] / sd; b = float(mdl.intercept_[0] - (mdl.coef_[0] * mu / sd).sum())
print('weights (per standard deviation): ' + '  '.join('%s %+.2f' % q for q in sorted(zip(level.PAN_NAMES, mdl.coef_[0]), key=lambda q: -abs(q[1]))))
best = sorted(((v, n, m) for (t, n, m), v in zip(tag, s) if t == 'clean'), key=lambda q: -q[0])[:12]
for v, n, m in best: print('  %-24s %5.2f | %.2f Hz swing %.1f adj %.2f resid %.2f slope %.2f [%.2f %.2f] cyc %.1f steep %.2f fund %.2f %s' % (n, v, FS / m['P'], m['swing'], m['adj'], m['resid'], m['slope_med'], m['slope_lo'], m['slope_hi'], m['cycles'], m['steep'], m['fund'], 'slow' if m['slow'] else ''))
if '--write' in sys.argv:
    far = float(sys.argv[sys.argv.index('--write') + 1]); p = '/tmp/claude-0/v2/level/rule.json'; R = json.load(open(p)) if os.path.exists(p) else {}
    R['pan'] = dict(names=level.PAN_NAMES, w=[float(v) for v in w], b=b, cut=cuts[far], far=far); json.dump(R, open(p, 'w'), indent=1); print('written: pan, cut %.3f at %.1f%%' % (cuts[far], 100 * far))
