# A conjunctive rule ("tight agreement in every respect") against the fitted sum: thresholds searched so that the most true movements
# are found while at most a few clean clips (train + corpus dev) pass.     python3 fit6.py
import json, sys, numpy as np, collections
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level
GATED = {'tr_synthetic_005', 'tr_synthetic_055'}
rows = [r for r in (json.loads(l) for l in open('data/stage2.jsonl')) if r['name'] not in GATED]
rows += [dict(name=r['name'], clean=r['tog'], wet=[], par=dict(kind='none'), dev=True) for r in (json.loads(l) for l in open('data/corpus_dev.jsonl'))]
ok = lambda m: level.first_filter(m) and not level.struck(m)
KEYS = ['slope_spread', 'slope_off', 'cyc_spread', 'cyc_off', 'halves', 'corr_lo', 'dis_hi', 'excess', 'slope_range']
def vec(m): return [m['slope_hi'] - m['slope_lo'], abs(m['slope_med'] - 1.0), m['cyc_hi'] - m['cyc_lo'], abs(m['cyc_med'] - 1.0), -m['halves'], -m['corr_lo'], m['dis_hi'], -m['excess'], m['slope_max'] - m['slope_min']]      # all: smaller is better
clean = [(r['name'], np.array([vec(m) for m in r['clean'] if ok(m)]).reshape(-1, len(KEYS)), bool(r.get('dev'))) for r in rows]
wet = [(r['name'], r['par'], np.array([vec(m) for m in r['wet'] if ok(m) and m['match'] == 1]).reshape(-1, len(KEYS))) for r in rows if r['par']['kind'] in ('trem', 'gate')]
print('true-period candidates, measures at percentiles 50 75 90 95 (smaller is better):')
for kind in ('trem', 'gate'):
    A = np.concatenate([v for n, p, v in wet if p['kind'] == kind and len(v)]); print('  %s: ' % kind + '  '.join('%s %s' % (k, np.round(np.percentile(A[:, i], [50, 75, 90, 95]), 2)) for i, k in enumerate(KEYS)))
A = np.concatenate([v for n, v, d in clean if len(v)]); print('  clean: ' + '  '.join('%s %s' % (k, np.round(np.percentile(A[:, i], [5, 10, 25, 50]), 2)) for i, k in enumerate(KEYS)))
def evaluate(th):
    nc = sum(1 for n, v, d in clean if len(v) and (v <= th).all(1).any()); nd = sum(1 for n, v, d in clean if d and len(v) and (v <= th).all(1).any())
    ft = sum(1 for n, p, v in wet if p['kind'] == 'trem' and len(v) and (v <= th).all(1).any()); fg = sum(1 for n, p, v in wet if p['kind'] == 'gate' and len(v) and (v <= th).all(1).any()); return nc, nd, ft, fg
rng = np.random.default_rng(0); best = {}
lo = np.array([0.10, 0.05, 0.15, 0.05, -0.99, -0.99, 0.04, -0.60, 0.2]); hi = np.array([1.00, 1.00, 2.00, 1.00, -0.50, -0.60, 1.00, -0.02, 3.0])
for it in range(60000):
    th = lo + (hi - lo) * rng.random(len(KEYS)) ** 1.5; nc, nd, ft, fg = evaluate(th)
    for cap in (2, 4, 6):
        if nc <= cap and nd <= 1 and (cap not in best or ft + fg > best[cap][0]): best[cap] = (ft + fg, ft, fg, nc, nd, th.copy())
nt = sum(1 for n, p, v in wet if p['kind'] == 'trem'); ng = sum(1 for n, p, v in wet if p['kind'] == 'gate')
for cap, (tot, ft, fg, nc, nd, th) in sorted(best.items()):
    print('at most %d clean of %d (got %d, corpus dev %d): trem %d/%d (%.0f%%) gate %d/%d (%.0f%%) | ' % (cap, len(clean), nc, nd, ft, nt, 100 * ft / nt, fg, ng, 100 * fg / ng) + '  '.join('%s<=%.2f' % (k, t) for k, t in zip(KEYS, th)))
    print('    clean passing: ' + ' '.join(n for n, v, d in clean if len(v) and (v <= th).all(1).any()))
