# The rule "is this lag an echo", third round: the measures of echo2.measure (stricter copy-level measure), a small boosted model.
# Data: mkdata2.py (1057 clean 6 s clips with and without a synthetic echo).
#   python3 fit3.py [save]     prints out-of-fold rates per clip at cuts chosen by the false-find rate on clean clips; "save" fits on all and writes rule3.pkl
import json, sys, glob, numpy as np, collections, pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, '.'); import echo2
rows = []
for f in sorted(glob.glob('d2_*.jsonl')):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
def make(): return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.06, l2_regularization=1.0, min_samples_leaf=12)
X = []; y = []; grp = []; meta = []
for i, r in enumerate(rows):
    D = r['par']['ms']
    for k, c in enumerate(r['wet']):
        ratio = c['ms'] / D
        if abs(ratio - 1) < 0.004: lab = 1
        elif round(ratio) >= 2 and abs(ratio - round(ratio)) < 0.004: continue          # a multiple of the echo: neither right nor wrong
        else: lab = 0
        X.append(echo2.feats(c)); y.append(lab); grp.append(i); meta.append((i, 'wet', k))
    for k, c in enumerate(r['clean']): X.append(echo2.feats(c)); y.append(0); grp.append(i); meta.append((i, 'clean', k))
X = np.array(X, float); y = np.array(y); grp = np.array(grp)
g = np.abs(np.array([r['par']['g'] for r in rows])); D = np.array([r['par']['ms'] for r in rows]); fb = np.array([r['par']['fb'] for r in rows])
hit = np.array([any(abs(c['ms'] / r['par']['ms'] - 1) < 0.004 for c in r['wet']) for r in rows])
print('%d clips, %d candidates (%d true echoes, prior %.3f) | the echo is among the measured candidates on %.0f%%' % (len(rows), len(y), y.sum(), y.mean(), 100 * hit.mean()))
if __name__ == '__main__':
    p = np.zeros(len(y))
    for tr, te in GroupKFold(8).split(X, y, grp): p[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    np.save('oof3.npy', p)
    acc = collections.defaultdict(list)
    for j, (i, w, rank) in enumerate(meta): acc[(i, w)].append((rank, p[j], y[j]))
    cleanmax = np.array([max([pp for _, pp, _ in acc.get((i, 'clean'), [])] or [0.0]) for i in range(len(rows))])
    for policy in ('first', 'best'):
        print('\npolicy: %s' % ('the loudest candidate over the cut' if policy == 'first' else 'the candidate with the highest p, if over the cut'))
        for mf in (0.0, 0.2, 0.5, 1.0, 2.0):
            cut = np.sort(cleanmax)[::-1][int(round(mf / 100 * len(rows)))] + 1e-9 if mf > 0 else cleanmax.max() + 1e-9
            found = np.zeros(len(rows), bool); wrong = np.zeros(len(rows), bool)
            for i in range(len(rows)):
                cw = sorted(acc.get((i, 'wet'), []))
                if policy == 'first': pick = next(((pp, yy) for _, pp, yy in cw if pp >= cut), None)
                else:
                    pick = max(((pp, yy) for _, pp, yy in cw), default=None)
                    if pick is not None and pick[0] < cut: pick = None
                if pick is not None: found[i] = pick[1] == 1; wrong[i] = pick[1] == 0
            s = '  false finds %.1f%% (cut %.3f): echo found %4.1f%%, wrong lag %4.1f%% |' % (100 * (cleanmax >= cut).mean(), cut, 100 * found.mean(), 100 * wrong.mean())
            for lab, sel in (('g<0.3', g < 0.3), ('0.3-0.5', (g >= 0.3) & (g < 0.5)), ('g>=0.5', g >= 0.5)): s += ' %s %3.0f%%' % (lab, 100 * found[sel].mean())
            s += ' |'
            for lab, sel in (('<60ms', D < 60), ('60-250', (D >= 60) & (D < 250)), ('250-600', (D >= 250) & (D < 600)), ('>=600', D >= 600)): s += ' %s %3.0f%%' % (lab, 100 * found[sel].mean())
            s += ' | fb %3.0f%% single %3.0f%%' % (100 * found[fb > 0].mean(), 100 * found[fb == 0].mean())
            print(s)
    # calibration of the out-of-fold p
    print('\ncalibration (out of fold): ' + ' '.join('%.1f-%.1f: %.2f (n %d)' % (a, b, y[(p >= a) & (p < b)].mean() if ((p >= a) & (p < b)).any() else float('nan'), ((p >= a) & (p < b)).sum()) for a, b in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .97), (.97, 1.01))))
    if 'save' in sys.argv:
        m = make().fit(X, y); pickle.dump(m, open('rule3.pkl', 'wb')); print('rule3.pkl written (fitted on all %d candidates)' % len(y))
