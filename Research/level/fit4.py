# The rule "a shared level movement (tremolo, gate) is there", fitted on the second-look data (data/stage2.jsonl): candidates of clean
# 6 s train clips (label 0) and, on the same clips with one synthetic movement put on, the candidate with the true period (label 1).
# The cut is set on out-of-fold scores so that a given share of ALL clean clips has a candidate over it.   python3 fit4.py [--write FAR]
import json, sys, glob, numpy as np, collections
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level
GATED = {'tr_synthetic_005', 'tr_synthetic_055'}                           # noise gated on and off by construction (train/mkclips.py): a true gate, 18 dB deep, not a clean sound
rows = [r for r in (json.loads(l) for l in open('/tmp/claude-0/v2/level/data/stage2.jsonl')) if r['name'] not in GATED]; nclean = len(rows)
import os
DEV = '/tmp/claude-0/v2/level/data/corpus_dev.jsonl'; WDEV = 3.0                 # the dry corpus sounds of the dev split: more clean sounds (loops, real recordings), each counted three times in the fit
if os.path.exists(DEV) and '--nodev' not in sys.argv: rows += [dict(name=r['name'], clean=r['tog'], wet=[], par=dict(kind='none'), dev=True) for r in (json.loads(l) for l in open(DEV))]; nclean = len(rows)
isdev = {r['name'] for r in rows if r.get('dev')}
ok = lambda m: level.first_filter(m) and not level.struck(m)
X, y, grp, tag = [], [], [], []
for r in rows:
    for m in r['clean']:
        if ok(m): X.append(level.rule_features(m)); y.append(0); grp.append(r['name']); tag.append(('clean', r['name'], m))
    for m in r['wet']:
        if not ok(m): continue
        if m['match'] == 1: X.append(level.rule_features(m)); y.append(1); grp.append(r['name']); tag.append(('wet', r['name'], m))
        elif m['match'] == 0: X.append(level.rule_features(m)); y.append(0); grp.append(r['name']); tag.append(('wetoff', r['name'], m))
X = np.array(X, float); X[~np.isfinite(X)] = 0.0; y = np.array(y); grp = np.array(grp); mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd
print('%d clips; candidates past the first filter and not struck: %d true, %d clean, %d off-period on clips with a movement' % (nclean, y.sum(), sum(t[0] == 'clean' for t in tag), sum(t[0] == 'wetoff' for t in tag)))
st = collections.Counter(); 
for r in rows:
    for m in r['wet']:
        if m['match'] == 1: k = r['par']['kind'] + ' ' + r['par'].get('shape', ''); st[k, 'n'] += 1; st[k, 'first'] += level.first_filter(m); st[k, 'struck'] += level.first_filter(m) and level.struck(m)
print('true-period candidates: ' + '; '.join('%s %d (past first filter %d, of those struck %d)' % (k, st[k, 'n'], st[k, 'first'], st[k, 'struck']) for k in sorted({k for k, _ in st})))
print('clean candidates past the first filter: %d, of those struck: %d' % (sum(level.first_filter(m) for r in rows for m in r['clean']), sum(level.first_filter(m) and level.struck(m) for r in rows for m in r['clean'])))
C = 0.3; sw = np.array([WDEV if g in isdev else 1.0 for g in grp])
def oof():
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Z, y, grp): s[te] = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(Z[tr], y[tr], sample_weight=sw[tr]).decision_function(Z[te])
    return s
s = oof(); bc = collections.defaultdict(lambda: -1e9); cand = collections.defaultdict(list)
for (t, n, m), v in zip(tag, s):
    if t == 'clean': bc[n] = max(bc[n], v)
    else: cand[n].append((v, m))
a = np.sort(np.array([bc[r['name']] for r in rows]))[::-1]; par = {r['name']: r['par'] for r in rows}; cuts = {}
for far in (0.002, 0.004, 0.005, 0.008, 0.01, 0.02, 0.05):
    k0 = int(np.floor(far * len(a))); cut = float(a[k0]) + 1e-9; cuts[far] = cut; got = collections.Counter(); tot = collections.Counter(); wrong = 0; nrep = int((a >= cut).sum())
    for r in rows:
        p = r['par']
        if p['kind'] not in ('trem', 'gate'): continue
        ks = [p['kind'], p['kind'] + (' deep' if p['depth'] >= 0.6 else ' shallow')] + ([p['kind'] + ' ' + p['shape']] if 'shape' in p else []) + (['gate pattern' if p['pattern'] != 0xFFFF else 'gate plain'] if p['kind'] == 'gate' else [])
        for k in ks: tot[k] += 1
        acc = [(v, m) for v, m in cand[r['name']] if v >= cut]
        if not acc: continue
        v, m = max(acc, key=lambda q: q[0])
        for v2, m2 in sorted(acc, key=lambda q: q[1]['P']):
            rr = m['P'] / m2['P']
            if m2['P'] < m['P'] * 0.75 and abs(rr - round(rr)) < 0.03 * round(rr) and m2['share'] >= 0.6 * m['share']: m = m2; break
        if m['match'] == 1:
            for k in ks: got[k] += 1
        else: wrong += 1
    ndev = sum(1 for r in rows if r.get('dev') and bc[r['name']] >= cut)
    print('clean clips reported %d of %d, of the corpus dev sounds %d of %d (cut %.2f): ' % (nrep, len(a), ndev, len(isdev), cut) + '  '.join('%s %d/%d (%.0f%%)' % (k, got[k], tot[k], 100 * got[k] / max(tot[k], 1)) for k in sorted(tot)) + '  | reported with a wrong period: %d' % wrong)
mdl = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(Z, y, sample_weight=sw); w = mdl.coef_[0] / sd; b = float(mdl.intercept_[0] - (mdl.coef_[0] * mu / sd).sum())
print('weights (per standard deviation): ' + '  '.join('%s %+.2f' % q for q in sorted(zip(level.RULE_NAMES, mdl.coef_[0]), key=lambda q: -abs(q[1]))))
best = {}
for (t, n, m), v in zip(tag, s):
    if t == 'clean' and (n not in best or v > best[n][0]): best[n] = (v, m)
print('clean clips with the highest scores (out of fold):')
for n, (v, m) in sorted(best.items(), key=lambda q: -q[1][0])[:22]:
    print('  %-24s %5.2f | %.2f Hz depth %5.1f cyc %2d corr_lo %.2f slopes [%.2f %.2f] cycsize [%.2f %.2f] halves %.2f share %.2f fall %.2f climb %.2f flat %.2f dis %.2f/%.2f steep %.2f' % (n, v, 1000 / m['P'], m['depth'], m['cyc_n'], m['corr_lo'], m['slope_lo'], m['slope_hi'], m['cyc_lo'], m['cyc_hi'], m['halves'], m['share'], m['fall'], m['climb'], m['flat'], m['dis_med'], m['dis_hi'], m['steep']))
if '--write' in sys.argv:
    far = float(sys.argv[sys.argv.index('--write') + 1]); import os
    p = '/tmp/claude-0/v2/level/rule.json'; R = json.load(open(p)) if os.path.exists(p) else {}
    R['together'] = dict(names=level.RULE_NAMES, w=[float(v) for v in w], b=b, cut=cuts[far], far=far); json.dump(R, open(p, 'w'), indent=1); print('written: together, cut %.3f at %.1f%%' % (cuts[far], 100 * far))
