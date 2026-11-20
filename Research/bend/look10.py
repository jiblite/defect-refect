# The clean sounds the early meter is surest about: are they bent after all? Figures only, kinds only, never names.
import os, pickle, sys
from common import *
import bfeat, bengine, bmeter
M = bmeter.model(sys.argv[1]); cache = pickle.load(open('data/beval_X.pkl', 'rb')); rows = []
for p in sorted(glob.glob(CORPUS + 'dry/*')):
    X, vp = cache['clean|' + os.path.basename(p)]; S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); rows.append((logit, p, S))
def figures(p):
    x = sf.read(p, dtype='float64', always_2d=True)[0]; pk = bfeat.peak_of(x); a = np.abs(x).ravel() / pk; db = 20 * np.log10(np.maximum(a, 1e-9))
    hist = [100 * np.mean((db <= -lo) & (db > -lo - 0.5)) for lo in np.arange(0, 6, 0.5)]; over = 100 * np.mean(db > 0)
    d = np.abs(np.diff(x[:, 0])) / pk; hi = np.abs(x[:-1, 0]) > 0.5 * pk; flat = 100 * np.mean(d[hi] < 0.002) if hi.sum() else 0.0
    crest = 20 * np.log10(np.abs(x).max() / np.sqrt(np.mean(x ** 2)))
    f, t, Z = ss.stft(x[:, 0], 48000, 'hann', 8192, 2048, boundary=None, padded=False); P = (np.abs(Z) ** 2).mean(1)[4:]; sfm = 10 * np.log10(np.exp(np.mean(np.log(P + 1e-30))) / np.mean(P)); fpk = f[4:][np.argmax(P)]
    fr = np.lib.stride_tricks.sliding_window_view(np.abs(x[:, 0]), 2048)[::1024].max(1); lv = 20 * np.log10(np.maximum(fr, 1e-9) / pk); spread = np.percentile(lv, 90) - np.percentile(lv, 10)
    return crest, over, hist, flat, sfm, fpk, spread, len(x) / 48000
rows.sort(key=lambda r: -r[0]); print('the six surest of', len(rows))
for logit, p, S in rows[:6]:
    c, over, h, flat, sfm, fpk, spread, dur = figures(p)
    print('%-4s sure %4.1f | crest %4.1f dB | over the peak %.3f %% | share per half dB from the peak down: %s | flat among the large values %.1f %% | tonal %.0f dB, strongest at %.0f Hz | frame peaks spread %.1f dB | %.1f s | squeeze read %s' % (
          kind(os.path.basename(p)), logit, c, over, ' '.join('%.1f' % v for v in h), flat, sfm, fpk, spread, dur, ' '.join('%.0f' % v for v in np.clip(S[0], 0, 40))))
allf = [figures(p) for _, p, _ in rows[6::8]]
print('for comparison, 24 other clean sounds: crest median %.1f | flat median %.1f %%, most %.1f %% | spread median %.1f dB, least %.1f | share in the top 3 dB median %.1f %%, most %.1f %%' % (np.median([f[0] for f in allf]), np.median([f[3] for f in allf]), max(f[3] for f in allf),
      np.median([f[6] for f in allf]), min(f[6] for f in allf), np.median([sum(f[2][:6]) for f in allf]), max(sum(f[2][:6]) for f in allf)))
