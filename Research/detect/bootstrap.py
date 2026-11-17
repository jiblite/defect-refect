# How firm the held-out numbers are: resampling the held-out sounds (clean: by source clip; effects: within each family).     python3 bootstrap.py VARIANT
import sys
import paths
import numpy as np, verifier as V, vreport as R
v = sys.argv[1]; te = V.test_rows(); z = np.load(V.D + 'data/test_%s.npz' % v); B = z['S'] > z['T']; A = R.before(te); Y = np.array([[f in r['labels'] for f in V.FAMS] for r in te])
held = np.array([r['set'] == 'test' for r in te]); main = np.flatnonzero(np.array([r['variant'] in ('dry', 'case_reverb', 'case_tone') for r in te]) & held); rng = np.random.default_rng(0)
src = np.array([te[i]['group'] for i in main]); us = sorted(set(src)); idx = { s: main[src == s] for s in us }; c = []; f = []
for _ in range(4000):
    pick = np.concatenate([idx[us[j]] for j in rng.integers(0, len(us), len(us))]); n = B[pick].sum(1); c.append(np.mean(n == 0)); f.append(n.mean())
print('held-out clean sounds: %d from %d source clips; verifier clean %.0f %% (95 %% range %.0f - %.0f), false effects per sound %.3f (%.3f - %.3f); false finds in all: %d' % (len(main), len(us), 100 * np.mean(B[main].sum(1) == 0),
      100 * np.percentile(c, 2.5), 100 * np.percentile(c, 97.5), B[main].sum(1).mean(), np.percentile(f, 2.5), np.percentile(f, 97.5), B[main].sum()))
pos = [np.flatnonzero(held & Y[:, k]) for k in range(len(V.FAMS))]; d = []; a_ = []; b_ = []
for _ in range(4000):
    ra = []; rb = []
    for k, p in enumerate(pos): q = p[rng.integers(0, len(p), len(p))]; ra.append(A[q, k].mean()); rb.append(B[q, k].mean())
    d.append(np.mean(rb) - np.mean(ra)); a_.append(np.mean(ra)); b_.append(np.mean(rb))
print('held-out effects: %d sounds; found before %.1f %% (%.0f - %.0f), verifier %.1f %% (%.0f - %.0f); verifier minus before %.1f points (95 %% range %.1f to %.1f)' % (sum(len(p) for p in pos), 100 * np.mean([A[p, k].mean() for k, p in enumerate(pos)]),
      100 * np.percentile(a_, 2.5), 100 * np.percentile(a_, 97.5), 100 * np.mean([B[p, k].mean() for k, p in enumerate(pos)]), 100 * np.percentile(b_, 2.5), 100 * np.percentile(b_, 97.5), 100 * (np.mean([B[p, k].mean() for k, p in enumerate(pos)]) - np.mean([A[p, k].mean() for k, p in enumerate(pos)])), 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)))
