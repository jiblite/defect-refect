# The shipped echo finder (tone.find_echoes via tone.contour) on clean training clips with one reflection put on by hand:
# how often is it found, and how exact are delay and gain? One process.
import sys, os, glob, json, time
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/tmp/claude-0/v2/tone')
import numpy as np, soundfile as sf, tone
def reflect(x, d, g):
    n = len(x) + 2048; X = np.fft.rfft(x, n, axis=0); w = 2 * np.pi * np.fft.rfftfreq(n) * d; return np.fft.irfft(X * (1 + g * np.exp(-1j * w))[:, None], n, axis=0)[:len(x)]
if __name__ == '__main__':
    files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac')); r = np.random.default_rng(7); files = [files[i] for i in r.permutation(len(files))[:120]]; t0 = time.time(); R = {0.3: [], 0.45: [], 0.7: []}
    for p in files:
        x = sf.read(p, always_2d=True)[0].astype(np.float64)
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        for g0 in R:
            d = float(np.exp(r.uniform(np.log(0.6 * 48), np.log(12 * 48)))); g = g0 * float(r.choice([-1, 1])); info = []; tone.contour(reflect(x[:, :2], d, g), info=info); e = info[0]['echoes']
            hit = [(dd, gg) for dd, gg in e if abs(dd - d) <= 0.5]; R[g0].append((1.0 if hit else 0.0, abs(hit[0][0] - d) if hit else np.nan, abs(hit[0][1] - g) if hit else np.nan, float(len(e) - len(hit)), d))
    print('%d clean clips, one reflection each (delay 0.6..12 ms, either sign), %.0f s' % (len(files), time.time() - t0))
    for g0, v in R.items():
        v = np.array(v); short = v[:, 4] < 96
        print('  gain %.2f: found on %.0f%% (delays under 2 ms: %.0f%%, longer: %.0f%%) | delay error median %.3f samples, 90th percentile %.3f | gain error median %.3f, 90th percentile %.3f | other reflections reported: %d' % (g0, 100 * v[:, 0].mean(), 100 * v[short, 0].mean(), 100 * v[~short, 0].mean(), np.nanmedian(v[:, 1]), np.nanpercentile(v[:, 1], 90), np.nanmedian(v[:, 2]), np.nanpercentile(v[:, 2], 90), int(v[:, 3].sum())))
    json.dump({str(k): v for k, v in R.items()}, open('/tmp/claude-0/v2/tone/lab/work/echo_sim.json', 'w'))
