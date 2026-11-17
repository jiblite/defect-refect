# Trying forms of the rule: transformed measures (what matters is how close to perfect the agreement is), hand rule for comparison.
import json, sys, glob, numpy as np, collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level
from fit2 import rows, match
def nl(v, e=0.02): return -np.log(max(1.0 - v, 0.0) + e)                       # near 1 is what counts
def feats(m):
    sp = m['slope_hi'] - m['slope_lo']
    return [np.log(max(m['depth'], 0.3)), nl(m['corr_lo']), nl(m['corr_med']), nl(m['corr_min']), np.log(min(sp, 3.0) + 0.03), np.log(min(abs(m['slope_med'] - 1.0), 2.0) + 0.03), np.log(min(max(m['slope_max'] - m['slope_min'], 0.0), 6.0) + 0.05),
            np.log(min(abs(m['cyc_med'] - 1.0), 2.0) + 0.03), np.log(min(max(m['cyc_hi'] - m['cyc_lo'], 0.0), 4.0) + 0.03), nl(m['halves']), nl(max(m['r2c'], -1.0), 0.05), nl(m['share'], 0.05), nl(m['excess'], 0.05), np.log(max(m['cyc_n'], 1)), m['steep'], nl(m['fund'], 0.05), m['silent'], m['skipfrac'], nl(max(m['r2'], -1.0), 0.05)]
NAMES = ['logdepth', 'corr_lo', 'corr_med', 'corr_min', 'slope_spread', 'slope_off', 'slope_range', 'cyc_off', 'cyc_spread', 'halves', 'r2c', 'share', 'excess', 'logcyc', 'steep', 'fund', 'silent', 'skipfrac', 'r2']
def build(feats):
    X, y, grp, tag = [], [], [], []
    for r in rows:
        for m in r['clean']['tog']: X.append(feats(m)); y.append(0); grp.append(r['name']); tag.append(('clean', r['name'], m))
        if r['par']['kind'] in ('trem', 'gate'):
            for m in r['wet']['tog']:
                mt = match(m['P'], r['par']['rate'])
                if mt == 1: X.append(feats(m)); y.append(1); grp.append(r['name']); tag.append(('wet', r['name'], m))
                elif mt == 0: X.append(feats(m)); y.append(0); grp.append(r['name']); tag.append(('wetoff', r['name'], m))
    X = np.array(X, float); X[~np.isfinite(X)] = 0.0; return X, np.array(y), np.array(grp), tag
def view(s, tag, label, fars=(0.005, 0.01, 0.02, 0.05)):
    bc = collections.defaultdict(lambda: -1e9); bt = collections.defaultdict(lambda: -1e9)
    for (t, n, m), v in zip(tag, s):
        if t == 'clean': bc[n] = max(bc[n], v)
        if t == 'wet': bt[n] = max(bt[n], v)
    a = np.sort(np.array([bc[r['name']] for r in rows]))[::-1]; print(label); cuts = {}
    for far in fars:
        cut = float(a[int(np.floor(far * len(a)))]) + 1e-9; got = collections.Counter(); tot = collections.Counter(); cuts[far] = cut
        for r in rows:
            k = r['par']['kind']
            if k not in ('trem', 'gate'): continue
            d = 'deep' if r['par']['depth'] >= 0.6 else 'shallow'; tot[k] += 1; tot[k + ' ' + d] += 1
            if bt[r['name']] >= cut: got[k] += 1; got[k + ' ' + d] += 1
        print('   clean over the cut %.1f%% (cut %6.2f): ' % (100 * far, cut) + '  '.join('%s %d/%d (%.0f%%)' % (k, got[k], tot[k], 100 * got[k] / max(tot[k], 1)) for k in sorted(tot)))
    return cuts, bc
def oof(make, Z, y, grp):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Z, y, grp):
        s[te] = make().fit(Z[tr], y[tr]).decision_function(Z[te])
    return s
if __name__ == '__main__':
    X, y, grp, tag = build(feats); mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd; print(len(rows), 'clips', len(y), 'candidates', y.sum(), 'true')
    for C in (0.03, 0.3, 3.0):
        s = oof(lambda: LogisticRegression(C=C, max_iter=5000, class_weight='balanced'), Z, y, grp); cuts, bc = view(s, tag, 'logistic on transformed measures, C=%g:' % C)
    sg = oof(lambda: GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.06, subsample=0.8, random_state=0), Z, y, grp); view(sg, tag, 'boosted stumps-of-2:')
    h = np.array([1.0 if level.accept_together(m) else 0.0 for (t, n, m) in tag]); bc = collections.defaultdict(float); bt = collections.defaultdict(float)
    for (t, n, m), v in zip(tag, h):
        if t == 'clean': bc[n] = max(bc[n], v)
        if t == 'wet': bt[n] = max(bt[n], v)
    nt = [r for r in rows if r['par']['kind'] in ('trem', 'gate')]; print('hand rule: clean reported %d/%d (%.1f%%), true found %d/%d' % (sum(bc.values()), len(rows), 100 * sum(bc.values()) / len(rows), sum(bt[r['name']] for r in nt), len(nt)))
    mdl = LogisticRegression(C=0.3, max_iter=5000, class_weight='balanced').fit(Z, y); print('weights: ' + '  '.join('%s %+.2f' % q for q in sorted(zip(NAMES, mdl.coef_[0]), key=lambda q: -abs(q[1]))))
