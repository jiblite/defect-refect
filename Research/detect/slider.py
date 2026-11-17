# What the Sureness slider gives, from the held-out test sounds (and, for comparison, from the training sounds out of fold):
# the share of clean sounds that come out clean, the effects shown per clean sound that are not there, the share of effects
# found, and how often a shown effect of a given score is really there.     python3 slider.py VARIANT
import sys, pickle
import paths
import numpy as np, verifier as V, calib, export_verify as XV
variant = sys.argv[1]; D = paths.WORK; M = pickle.load(open(D + 'model/verifier_%s.pkl' % variant, 'rb')); names, fams = XV.read(D + 'model/verify_%s.bin' % variant)
z65 = np.array([f.z65 for f in fams]); slope = np.array([f.slope for f in fams]); FAMS = V.FAMS; assert [f.name for f in fams] == FAMS
def shown(X): return calib.shown(np.stack([m.decision_function(X) for m in M['models']], 1), z65, slope)
te = V.test_rows(); Xt, Yt, _ = V.matrices(te, variant); G = shown(Xt); held = np.array([r['set'] == 'test' for r in te]); var = np.array([r['variant'] for r in te])
main = held & np.isin(var, ['dry', 'case_reverb', 'case_tone']); clean = held & (Yt.sum(1) == 0); pos = held & (Yt.sum(1) > 0)
tr = V.train_rows(); Str = np.load(D + 'data/oof_%s.npz' % variant)['S']; Ytr = np.array([[f in r['labels'] for f in FAMS] for r in tr]); Gtr = calib.shown(calib.logit(Str), z65, slope); ctr = Ytr.sum(1) == 0
def rec(B, Y, sel): return 100 * np.mean([B[sel & Y[:, k], k].mean() for k in range(len(FAMS)) if (sel & Y[:, k]).any()])
print('HELD-OUT TEST SOUNDS (%d clean: %d dry, reverb only or tone only, the others the dry ones padded, as 16-bit, through a lossy codec, two mixed; %d with an effect)' % (clean.sum(), main.sum(), pos.sum()))
print('  slider | dry / reverb only / tone only: clean  false per sound | all clean sounds: clean  false per sound | effects found | other effects per sound with an effect | training sounds out of fold: clean  false  found')
for s in (0.30, 0.40, 0.50, 0.65, 0.80, 0.90, 0.99):
    B = G > s; Bt = Gtr > s
    print('  %4.0f %% | %34.0f %% %14.2f | %24.0f %% %14.2f | %11.0f %% | %38.2f | %32.0f %% %5.2f %5.0f %%' % (100 * s, 100 * np.mean(B[main].sum(1) == 0), B[main].sum(1).mean(), 100 * np.mean(B[clean].sum(1) == 0), B[clean].sum(1).mean(),
          rec(B, Yt, pos), (B[pos] & ~Yt[pos]).sum(1).mean(), 100 * np.mean(Bt[ctr].sum(1) == 0), Bt[ctr].sum(1).mean(), rec(Bt, Ytr, Ytr.sum(1) == 1)))
print('\nPER EFFECT at 50 / 65 / 80 %%: found on the held-out sounds that have it (n) | shown on held-out clean sounds (n = %d)' % clean.sum())
for k, f in enumerate(FAMS):
    p = pos & Yt[:, k]; print('  %-12s %3d | found %3.0f %% %3.0f %% %3.0f %% | on clean %4.1f %% %4.1f %% %4.1f %%' % ((f, p.sum()) + tuple(100 * (G[p, k] > s).mean() for s in (0.5, 0.65, 0.8)) + tuple(100 * (G[clean, k] > s).mean() for s in (0.5, 0.65, 0.8))))
print('\nHOW A SCORE READS: of the effects with a score in the range, on all %d held-out sounds, how many are really on the sound' % held.sum())
print('  (this depends on how common effects are: here %d of %d sounds carry one of the 17, each effect is on about %d of them)' % (pos.sum(), held.sum(), Yt[held].sum(0).mean()))
for lo, hi in ((0.30, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 0.95), (0.95, 1.01)):
    m = (G >= lo) & (G < hi) & held[:, None]; print('  %2.0f to %3.0f %%: %4d shown, %3.0f %% right' % (100 * lo, 100 * min(hi, 1.0), m.sum(), 100 * Yt[m].mean() if m.any() else 0))
m = (G > 0.65) & held[:, None]; print('  everything above 65 %%: %d shown, %.0f %% right' % (m.sum(), 100 * Yt[m].mean()))
