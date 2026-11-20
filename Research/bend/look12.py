# The clean sounds with the largest squeeze read at the peak: figures only, kinds only, never names.
import os, pickle, sys
from common import *
import bfeat, bengine, bmeter
M = bmeter.model(sys.argv[1]); cache = pickle.load(open('data/beval_X.pkl', 'rb')); rows = []
top = lambda S: max(bengine.tidy(S[0], bengine.SPREAD)[-1], bengine.tidy(S[1], bengine.SPREAD)[-1])
for p in sorted(glob.glob(CORPUS + 'dry/*')):
    X, vp = cache['clean|' + os.path.basename(p)]; S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); rows.append((top(S), logit, p))
rows.sort(key=lambda r: -r[0])
for t, logit, p in rows[:14]:
    x = sf.read(p, dtype='float64', always_2d=True)[0]; pk = bfeat.peak_of(x); a = np.abs(x).ravel() / pk; db = 20 * np.log10(np.maximum(a, 1e-9))
    hist = [100 * np.mean((db <= -lo) & (db > -lo - 0.5)) for lo in np.arange(0, 4, 0.5)]; crest = 20 * np.log10(np.abs(x).max() / np.sqrt(np.mean(x ** 2)))
    fr = np.lib.stride_tricks.sliding_window_view(np.abs(x[:, 0]), 2048)[::1024].max(1); lv = 20 * np.log10(np.maximum(fr, 1e-9) / pk); spread = np.percentile(lv, 90) - np.percentile(lv, 10)
    run = np.abs(np.diff(x[:, 0])) < 1e-4 * pk; hi = np.abs(x[:-1, 0]) > 0.9 * pk; flat = 100 * (run & hi).sum() / max(hi.sum(), 1)
    print('%-4s squeeze read %4.1f dB, logit %5.1f | crest %4.1f dB | share per half dB from the peak down: %s | of the values above 0.9 of the peak, standing still: %4.1f %% | frame peaks spread %4.1f dB' % (kind(os.path.basename(p)), t, logit, crest, ' '.join('%4.1f' % v for v in hist), flat, spread))
