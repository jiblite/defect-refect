# The REVERB sureness rule, fitted again for a new room model on the test sounds of prep_nw.py and prep_sb.py.
#   python3 fit_sure.py tag [FEATS=deep,bands] [PRIOR=0.62]
# Fitted on the 'dev' half (dev tools on dev clips), checked on the 'test' half (tools and clips the fit never saw).
# A sound counts as "with reverb" when a reverb tool ran on it (alone or after another effect), ringing out or cut off;
# "without" = clean clips, other effects only, tone effects only. The number is a probability for a sound that is
# PRIOR likely to carry reverb before anything is known: sounds with reverb weigh PRIOR / (their count), sounds
# without (1 - PRIOR) / (theirs), counted over ALL sounds, found or not; the rule itself only ranks the found ones.
import sys, os, json, collections, numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '/tmp/claude-0/v2/roomsure'); import look2
tag = sys.argv[1]; FEATS = os.environ.get('FEATS', 'deep,bands').split(','); PRIOR = float(os.environ.get('PRIOR', '0.62')); LOOK = os.environ.get('LOOK', 'as'); C = float(os.environ.get('C', '1.0'))
rows = look2.load('nw', tag) + look2.load('sb', tag)
def kind(r):
    if r.get('set') == 'nw': return {'fx': 'effect only', 'rev': 'reverb', 'fxr': 'effect + reverb'}[r['kind']] + (', cut off' if int(r.get('cut', 0)) else ', rings out' if r['kind'] != 'fx' else ', cut off')
    return {'clean': 'clean, cut off', 'reverb': 'reverb, rings out', 'tone': 'tone only'}.get(r['fam'], 'effect only, cut off')
col = dict(deep=lambda e: np.log10(e['deep'] + 0.003), bands=lambda e: e['bands'] / 41.0, moments=lambda e: np.log10(e['moments'] + 1.0), share=lambda e: e['share'] / 10.0, decay=lambda e: np.log2(max(e['decay'], 0.05)), strength=lambda e: e['strength'] / 10.0,
           own=lambda e: max(e['own'], -40.0) / 10.0, u3=lambda e: np.log10(e['u3'] + 0.003), u6=lambda e: np.log10(e['u6'] + 0.003), u20=lambda e: np.log10(e['u20'] + 0.003), r0=lambda e: e['r0'], r1=lambda e: e['r1'], r2=lambda e: e['r2'],
           end=lambda e: max(e.get('endDb', -60.0), -60.0) / 20.0, start=lambda e: max(e.get('startDb', -60.0), -60.0) / 20.0, live=lambda e: e['live'])
col['found'] = lambda e: float(e['found'])
def val(r, f):
    n, lk = f.split('@') if '@' in f else (f, LOOK); return col[n](r['j'][lk])          # 'bands@end': that number from the look with silence after the sound
def X_of(rs): return np.array([[val(r, f) for f in FEATS] for r in rs])
y = np.array([r['wet'] for r in rows]); dev = np.array([r['split'] == 'dev' for r in rows]); found = np.array([bool(r['j'][LOOK]['found']) for r in rows])
def fit(sel):
    """Rule fitted on the rows sel (found ones of them), with the prior weights counted over all of sel."""
    nw_, nd_ = y[sel].sum(), (1 - y[sel]).sum(); f = sel & found; w = np.where(y[f] == 1, PRIOR / nw_, (1 - PRIOR) / nd_) * sel.sum()
    m = LogisticRegression(C=C, max_iter=5000).fit(X_of([rows[i] for i in np.where(f)[0]]), y[f], sample_weight=w)
    fw, fd = (y[f] == 1).sum() / nw_, (y[f] == 0).sum() / nd_; none = PRIOR * (1 - fw) / (PRIOR * (1 - fw) + (1 - PRIOR) * (1 - fd)); return m, none, fw, fd
def sure(m, none, sel):
    s = np.full(len(rows), np.nan); f = sel & found; s[sel & ~found] = none
    if f.any(): s[f] = m.predict_proba(X_of([rows[i] for i in np.where(f)[0]]))[:, 1]
    return s
m, none, fw, fd = fit(dev); print('model pair %s | look "%s" | fitted on dev: %d with reverb (found on %.0f%%), %d without (found on %.0f%%)' % (tag, LOOK, y[dev].sum(), 100 * fw, (1 - y[dev]).sum(), 100 * fd))
print('rule: sureness = 1 / (1 + exp(-(%.3f %s)))   [%s]; no room found: %.2f' % (m.intercept_[0], ' '.join('%+.3f*%s' % (c, f) for c, f in zip(m.coef_[0], FEATS)), ', '.join(FEATS), none))
s = sure(m, none, np.ones(len(rows), bool)); ks = sorted(set(kind(r) for r in rows)); K = np.array([kind(r) for r in rows])
old = np.array([r['v_as'] if LOOK == 'as' else r['v_end'] for r in rows]); both = np.array([r['v'] for r in rows])
print('%-30s %-5s %4s  %6s | shown at 65%%: %8s %8s %8s | new rule at 50%% / 65%% / 80%%' % ('kind', 'split', 'n', 'found', 'old rule', 'old both', 'NEW'))
for k in ks:
    for sp, ss in (('dev', dev), ('test', ~dev)):
        q = (K == k) & ss
        if q.any(): print('%-30s %-5s %4d  %5.0f%% | %21.0f%% %7.0f%% %7.0f%% | %5.0f%% / %3.0f%% / %3.0f%%' % (k, sp, q.sum(), 100 * found[q].mean(), 100 * np.mean(old[q] > 0.65), 100 * np.mean(both[q] > 0.65), 100 * np.mean(s[q] > 0.65), 100 * np.mean(s[q] > 0.5), 100 * np.mean(s[q] > 0.65), 100 * np.mean(s[q] > 0.8)))
for sp, ss in (('dev', dev), ('test', ~dev)):
    w_ = ss & (y == 1); d_ = ss & (y == 0); print('%-4s all with reverb: shown %.0f%% (old rule %.0f%%, old both looks %.0f%%) | all without: %.0f%% (%.0f%%, %.0f%%)' % (sp, 100 * np.mean(s[w_] > 0.65), 100 * np.mean(old[w_] > 0.65), 100 * np.mean(both[w_] > 0.65), 100 * np.mean(s[d_] > 0.65), 100 * np.mean(old[d_] > 0.65), 100 * np.mean(both[d_] > 0.65)))
# honesty on the test half: what the numbers the rule says turn out to be (with the prior's weights)
t = ~dev; nw_, nd_ = y[t].sum(), (1 - y[t]).sum(); w = np.where(y == 1, PRIOR / nw_, (1 - PRIOR) / nd_)
print('honesty on the test half ("really" = share with reverb at the prior):')
for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.01)):
    q = t & (s >= lo) & (s < hi)
    if q.any(): print('  says %2.0f..%3.0f%%: n=%4d (%4d with, %4d without), mean said %.2f, really %.2f' % (100 * lo, 100 * min(hi, 1), q.sum(), y[q].sum(), (1 - y[q]).sum(), np.average(s[q], weights=w[q]), np.average(y[q], weights=w[q])))
# what a ranking by the rule can do at all on the test half: reverbs shown when the cut is set to let through 5 / 10 / 15 % of the sounds without
neg = np.sort(s[t & (y == 0)])[::-1]
for fa in (0.05, 0.10, 0.15):
    cut = neg[int(fa * len(neg))]; print('  cut where %2.0f%% of the test sounds without reverb pass (sureness %.2f): ringing %3.0f%%, cut off %3.0f%% of those with' % (100 * fa, cut, 100 * np.mean(s[t & (y == 1) & np.array(['rings' in k for k in K])] > cut), 100 * np.mean(s[t & (y == 1) & np.array(['cut off' in k for k in K])] > cut)))
