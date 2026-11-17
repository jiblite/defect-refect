# How should the cuts be set from the out-of-fold scores? Checked inside the training sounds only: the cuts are chosen on the
# sources of some folds and judged on the sources of the others (each half in turn), for several budgets (false effects per
# clean sound aimed at) and floors (the lowest cut allowed).        python3 pickcut.py VARIANT
import sys
import paths
import numpy as np, verifier as V
rows = V.train_rows(); S = np.load(V.D + 'data/oof_%s.npz' % sys.argv[1])['S']; assert len(S) == len(rows)
Y = np.array([[f in r['labels'] for f in V.FAMS] for r in rows]); clean = Y.sum(1) == 0; fold = np.array([r['fold'] for r in rows])
own = np.array([len(r['labels']) > 0 and all(i == 'own' for i in r['impl'][:len(r['labels'])]) for r in rows]); single = Y.sum(1) == 1
print('budget floor | chosen on one part, judged on the other: clean  false/sound  found own  found outside (unseen) | on all (optimistic): false/sound  found own  outside')
for b in (0.03, 0.05, 0.07, 0.10):
    for fl in (0.0, 0.1, 0.2, 0.3, 0.5):
        res = []
        for A in ((0, 1), (2, 3), (4, 0), (1, 2), (3, 4)):
            a = np.isin(fold, A); T, _ = V.thresholds(S[~a], Y[~a], clean[~a], b, fl); B = S > T
            rec = lambda sel: np.mean([B[sel & Y[:, k], k].mean() for k in range(len(V.FAMS)) if (sel & Y[:, k]).any()])
            res.append((np.mean(B[a & clean].sum(1) == 0), B[a & clean].sum(1).mean(), rec(a & single & own), rec(a & single & ~own)))
        r = np.mean(res, 0); T, _ = V.thresholds(S, Y, clean, b, fl); B = S > T; rec = lambda sel: np.mean([B[sel & Y[:, k], k].mean() for k in range(len(V.FAMS)) if (sel & Y[:, k]).any()])
        print('  %.2f  %.1f  | %38.0f %% %10.3f %8.0f %% %12.0f %% | %30.3f %8.0f %% %6.0f %%' % (b, fl, 100 * r[0], r[1], 100 * r[2], 100 * r[3], B[clean].sum(1).mean(), 100 * rec(single & own), 100 * rec(single & ~own)))
