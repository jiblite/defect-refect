# Headroom check for the together rule: boosted trees and extra hinge terms against the plain logistic (out of fold, same clips).
import json, sys, numpy as np, collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level
rows = [json.loads(l) for l in open('data/stage2.jsonl')]
GATED = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else set()
ok = lambda m: level.first_filter(m) and not level.struck(m)
def extra(m): return [max(0.0, 0.7 - m['cyc_lo']), max(0.0, m['cyc_hi'] - 1.3), max(0.0, 12 - m['cyc_n']) / 8.0, min(m['cyc_n'], 60) / 60.0, max(0.0, m['slope_hi'] - 1.2), max(0.0, 0.8 - m['slope_lo']), max(0.0, 0.9 - m['halves']), max(0.0, 0.3 - m['share']), max(0.0, 0.9 - m['corr_lo'])]
def build(fe):
    X, y, grp, tag = [], [], [], []
    for r in rows:
        if r['name'] in GATED: continue
        for m in r['clean']:
            if ok(m): X.append(fe(m)); y.append(0); grp.append(r['name']); tag.append(('clean', r['name'], m))
        for m in r['wet']:
            if not ok(m): continue
            if m['match'] == 1: X.append(fe(m)); y.append(1); grp.append(r['name']); tag.append(('wet', r['name'], m))
            elif m['match'] == 0: X.append(fe(m)); y.append(0); grp.append(r['name']); tag.append(('wetoff', r['name'], m))
    X = np.array(X, float); X[~np.isfinite(X)] = 0.0; return X, np.array(y), np.array(grp), tag
def run(fe, make, label, std=True):
    X, y, grp, tag = build(fe); Z = (X - X.mean(0)) / (X.std(0) + 1e-9) if std else X; s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Z, y, grp):
        mdl = make().fit(Z[tr], y[tr]); s[te] = mdl.decision_function(Z[te])
    bc = collections.defaultdict(lambda: -1e9); bt = collections.defaultdict(lambda: -1e9)
    for (t, n, m), v in zip(tag, s):
        if t == 'clean': bc[n] = max(bc[n], v)
        if t == 'wet': bt[n] = max(bt[n], v)
    names = [r['name'] for r in rows if r['name'] not in GATED]; a = np.sort(np.array([bc[n] for n in names]))[::-1]; print(label)
    for far in (0.005, 0.01, 0.02):
        cut = float(a[int(np.floor(far * len(a)))]) + 1e-9; got = collections.Counter(); tot = collections.Counter()
        for r in rows:
            p = r['par']
            if p['kind'] not in ('trem', 'gate') or r['name'] in GATED: continue
            ks = [p['kind']] + ([p['kind'] + ' no sawdown'] if p.get('shape', 'sawdown') != 'sawdown' else [])
            for k in ks: tot[k] += 1; got[k] += bt[r['name']] >= cut
        print('   clean reported %.1f%%: ' % (100 * far) + '  '.join('%s %d/%d (%.0f%%)' % (k, got[k], tot[k], 100 * got[k] / max(tot[k], 1)) for k in sorted(tot)) + '   top clean: ' + ' '.join('%s' % n.replace('tr_', '') for n, v in sorted(bc.items(), key=lambda q: -q[1])[:7]))
LR = lambda C=0.3: (lambda: LogisticRegression(C=C, max_iter=5000, class_weight='balanced'))
run(level.rule_features, LR(), 'logistic')
run(lambda m: level.rule_features(m) + extra(m), LR(), 'logistic + hinge terms')
run(lambda m: level.rule_features(m) + extra(m), LR(0.05), 'logistic + hinge terms, C=0.05')
run(lambda m: level.rule_features(m) + extra(m), lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.06, subsample=0.8, random_state=0), 'boosted trees', std=False)
run(lambda m: level.rule_features(m) + extra(m), lambda: GradientBoostingClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=1), 'boosted trees depth 2', std=False)
