# The app's rule for "how sure is it that there is a room on this sound", fitted on what sure_stats.py collected.
# The number is a probability under the assumption that a sound dropped on the app is as likely to carry reverb as not:
# sounds with reverb weigh 1 / (their count), sounds without 1 / (theirs), counted over ALL sounds (found or not), so
# the share of each kind the engine finds a room on at all is part of the answer. The rule only has to rank the found
# ones. Prints the rule, how honest its numbers are (cross-validated), and what the Sureness slider's 65 % then shows.
#   MODEL=<name> [FEATS=deep,bands] [C=1.0] python3 sure_fit2.py
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
MODEL = os.environ.get('MODEL', 'unet_b9'); rows = json.load(open('/tmp/claude-0/v2/train/sure_stats_%s.json' % MODEL)); C = float(os.environ.get('C', '1.0'))
feats = os.environ.get('FEATS', 'deep,bands').split(',')
col = dict(deep=lambda r: np.log10(r['deep'] + 0.003), bands=lambda r: r['evb'] / 41.0, moments=lambda r: np.log10(r['ev'] + 1.0), share=lambda r: r['share'] / 10.0, decay=lambda r: np.log2(r['tmid']), strength=lambda r: r['kmid'] / 10.0, wid=lambda r: r['wid'], gmean=lambda r: r['gmean'] / 10.0, evmid=lambda r: np.log10(r['evmid'] + 1.0), own=lambda r: max(r.get('own', -40.0), -40.0) / 10.0)
wet_all = np.array([r['wet'] for r in rows]); nw, nd = wet_all.sum(), (1 - wet_all).sum()
fi = np.array([i for i, r in enumerate(rows) if r['found']]); F = [rows[i] for i in fi]; y = wet_all[fi]
X = np.array([[col[f](r) for f in feats] for r in F]); w = np.where(y == 1, 0.5 / nw, 0.5 / nd) * len(rows)
fw, fd = y.sum() / nw, (1 - y).sum() / nd
print('model %s | %d sounds with reverb, %d without | a room is found on %.0f%% and %.0f%% | found alone says %.0f%% sure' % (MODEL, nw, nd, 100 * fw, 100 * fd, 100 * fw / (fw + fd)))
m = LogisticRegression(C=C, max_iter=2000).fit(X, y, sample_weight=w)
print('rule: sureness = 1 / (1 + exp(-(%.3f %s)))   [%s]' % (m.intercept_[0], ' '.join('%+.3f*%s' % (c, f) for c, f in zip(m.coef_[0], feats)), ', '.join(feats)))
# cross-validated numbers (several splits, averaged)
pc = np.zeros(len(y)); R = 10
for s in range(R):
    for tr, te in StratifiedKFold(8, shuffle=True, random_state=s).split(X, y):
        pc[te] += LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[te])[:, 1] / R
sure = np.zeros(len(rows)); sure[fi] = pc
PRIOR = float(os.environ.get('PRIOR', '0.5'))        # the app's assumption: this share of the sounds dropped on it carry reverb
if PRIOR != 0.5:
    sh = np.log(PRIOR / (1 - PRIOR)); sure[fi] = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + sh)))
    print('  with %.0f in 100 sounds carrying reverb: the rule\'s first number becomes %.3f, and no room found leaves %.2f' % (100 * PRIOR, m.intercept_[0] + sh, PRIOR * (1 - fw) / (PRIOR * (1 - fw) + (1 - PRIOR) * (1 - fd))))
ll = -np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / w.sum(); p0 = fw / (fw + fd); ll0 = -np.sum(w * (y * np.log(p0) + (1 - y) * np.log(1 - p0))) / w.sum()
print('  ranking among the found ones: log loss %.3f (found alone: %.3f)' % (ll, ll0))
print('  honesty (cross-validated; "really" = share with reverb at even odds):')
for lo, hi in [(0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.01)]:
    k = (pc >= lo) & (pc < hi)
    if k.sum(): print('    says %2.0f..%3.0f%%: n=%3d (%3d with, %2d without), mean said %.2f, really %.2f' % (100 * lo, 100 * min(hi, 1), k.sum(), y[k].sum(), (1 - y[k]).sum(), np.average(pc[k], weights=w[k]), np.average(y[k], weights=w[k])))
kind = lambda r: r.get('kind', '?')
def rate(sel, cut): return 100 * np.mean([sure[i] > cut for i in sel]) if len(sel) else float('nan')
I = range(len(rows)); wet = [i for i in I if rows[i]['wet']]; clear = [i for i in wet if rows[i]['drr'] < 8]; light = [i for i in wet if rows[i]['drr'] >= 8]
dry = [i for i in I if rows[i]['what'] == 'dry']; tone = [i for i in I if rows[i]['what'] == 'tone only']
loops = [i for i in I if not rows[i]['wet'] and kind(rows[i]) in ('sample-loops', 'sample-seq', 'loops', 'sequence')]; sure_dry = [i for i in I if not rows[i]['wet'] and kind(rows[i]) in ('speech-synth', 'synthetic', 'sample-notes')]
print('  kinds among the sounds without reverb:', sorted(set(kind(rows[i]) for i in I if not rows[i]['wet'])))
for cut in (0.3, 0.5, 0.65, 0.8, 0.9):
    print('  Sureness %2.0f%%: shown on %3.0f%% with reverb (clearly wet %3.0f%%, light %3.0f%%) | %3.0f%% of the dry | %3.0f%% of the tone-only | surely dry kinds %3.0f%%' % (100 * cut, rate(wet, cut), rate(clear, cut), rate(light, cut), rate(dry, cut), rate(tone, cut), rate(sure_dry, cut)))
if 'own' in rows[0]:
    for lab, sel in (('mono or narrow (under -20 dB unshared)', lambda r: r['own'] < -20), ('stereo', lambda r: r['own'] >= -20)):
        ww = [i for i in wet if sel(rows[i])]; dd = [i for i in I if not rows[i]['wet'] and sel(rows[i])]
        print('  %-40s with reverb n=%3d: found %3.0f%%, shown at 65%%: %3.0f%% | without n=%3d: found %3.0f%%, shown at 65%%: %3.0f%%, at 80%%: %3.0f%%' % (lab, len(ww), 100 * np.mean([rows[i]['found'] for i in ww]) if ww else 0, rate(ww, 0.65), len(dd), 100 * np.mean([rows[i]['found'] for i in dd]), rate(dd, 0.65), rate(dd, 0.8)))
print('  what found sounds are told (10/25/50/75/90%%): with reverb %s | without %s' % (' '.join('%.2f' % v for v in np.percentile(pc[y == 1], [10, 25, 50, 75, 90])), ' '.join('%.2f' % v for v in np.percentile(pc[y == 0], [10, 25, 50, 75, 90]))))
