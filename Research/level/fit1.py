# First look at the rule data: is the true period among the candidates, and what separates true movements from a clean sound's own rhythm.
import json, sys, glob, numpy as np, collections
rows = [json.loads(l) for f in sorted(glob.glob('data/train_*.jsonl')) for l in open(f)]
FS = 1000.0
def match(P, rate, tol=0.012):
    f = FS / P; r = f / rate
    if abs(r - 1) < tol: return 1
    for m in (2, 3, 4):
        if abs(r * m - 1) < tol: return m          # the candidate's period is m times the true one
    return 0
print(len(rows), 'clips', collections.Counter(r['par']['kind'] for r in rows))
for kind in ('trem', 'gate'):
    rs = [r for r in rows if r['par']['kind'] == kind]; hit = [next((i for i, m in enumerate(r['wet']['tog']) if match(m['P'], r['par']['rate']) == 1), -1) for r in rs]; hitm = [next((i for i, m in enumerate(r['wet']['tog']) if match(m['P'], r['par']['rate'])), -1) for r in rs]
    hit = np.array(hit); hitm = np.array(hitm); d = np.array([r['par']['depth'] for r in rs])
    print('%s: true period first %.0f%%, among the candidates %.0f%% (or a multiple of it: %.0f%%) | by depth <0.5: %.0f%%  0.5-0.75: %.0f%%  >0.75: %.0f%%' % (kind, 100 * (hit == 0).mean(), 100 * (hit >= 0).mean(), 100 * (hitm >= 0).mean(),
          100 * (hit[d < 0.5] >= 0).mean(), 100 * (hit[(d >= 0.5) & (d < 0.75)] >= 0).mean(), 100 * (hit[d >= 0.75] >= 0).mean()))
rs = [r for r in rows if r['par']['kind'] == 'pan']; ok = [r['wet']['pan'] is not None and min(abs(FS / r['wet']['pan']['P'] / r['par']['rate'] - 1), abs(FS / r['wet']['pan']['P'] / r['par']['rate'] / 2 - 1)) < 0.03 for r in rs]
print('pan: period right %.0f%% of %d' % (100 * np.mean(ok), len(rs)))
keys = ['share', 'excess', 'r2', 'r2c', 'depth', 'slope_med', 'slope_lo', 'slope_hi', 'corr_med', 'corr_lo', 'corr_min', 'cyc_med', 'cyc_lo', 'cyc_hi', 'halves', 'steep', 'fund', 'rise', 'up', 'silent', 'cycles']
pos = [m for r in rows if r['par']['kind'] in ('trem', 'gate') for m in r['wet']['tog'] if match(m['P'], r['par']['rate']) == 1]
neg = [max(r['clean']['tog'], key=lambda m: m['excess']) for r in rows if r['clean']['tog']]; negall = [m for r in rows for m in r['clean']['tog']]
print('%d true candidates, %d clean clips (their strongest candidate), %d clean candidates in all' % (len(pos), len(neg), len(negall)))
for k in keys:
    a = np.array([m[k] for m in pos]); b = np.array([m[k] for m in negall]); print('%-10s true: %6.2f %6.2f %6.2f %6.2f %6.2f | clean: %6.2f %6.2f %6.2f %6.2f %6.2f   (5 25 50 75 95%%)' % ((k,) + tuple(np.percentile(a, [5, 25, 50, 75, 95])) + tuple(np.percentile(b, [5, 25, 50, 75, 95]))))
kp = ['swing', 'share', 'adj', 'resid', 'slope_med', 'slope_lo', 'slope_hi', 'cycles', 'known', 'fund', 'steep']
pp = [r['wet']['pan'] for r in rows if r['par']['kind'] == 'pan' and r['wet']['pan']]; pn = [r['clean']['pan'] for r in rows if r['clean']['pan']]
print('pan: %d true, %d clean' % (len(pp), len(pn)))
for k in kp:
    a = np.array([m[k] for m in pp], float); b = np.array([m[k] for m in pn], float); print('%-10s true: %6.2f %6.2f %6.2f %6.2f %6.2f | clean: %6.2f %6.2f %6.2f %6.2f %6.2f' % ((k,) + tuple(np.percentile(a, [5, 25, 50, 75, 95])) + tuple(np.percentile(b, [5, 25, 50, 75, 95]))))
