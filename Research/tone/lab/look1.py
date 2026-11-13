import sys, glob, json, time
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, truth, score
SR = 48000
ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/tone/*/'))
t0 = time.time()
for d in ds:
    c = truth.load(d, 'A'); m = c['meta']; h = c['h']
    N = 1 << 17
    H = np.stack([np.fft.rfft(h[i][:N], N, axis=0) for i in range(2)])   # (in, bins, out)
    f = np.fft.rfftfreq(N, 1 / SR)
    e = np.array([[ (h[i][:, o] ** 2).sum() for o in range(2)] for i in range(2)])
    # length of the response: time until 99.9 % of energy
    hs = (h[0] ** 2).sum(1) + (h[1] ** 2).sum(1); ce = np.cumsum(hs) / hs.sum(); L999 = int(np.argmax(ce > 0.999)); L99999 = int(np.argmax(ce > 0.99999))
    band = (f > 50) & (f < 16000)
    mLL = 20 * np.log10(np.abs(H[0, :, 0]) + 1e-12); mRR = 20 * np.log10(np.abs(H[1, :, 1]) + 1e-12)
    dw = score.distance(c['dry'], c['wet'])
    print('%-22s %-4s fixed %6.1f  dw %5.2f  E LL %.3f LR %.3g RL %.3g RR %.3f  len99.9 %6.1f ms len99.999 %7.1f ms  LLrange %5.1f..%5.1f dB  L-R rms %.2f dB  gainA %.2f' % (
        c['name'], m['split'], m['fixed_db'], dw, e[0, 0], e[0, 1], e[1, 0], e[1, 1], L999 / 48, L99999 / 48, mLL[band].min(), mLL[band].max(), np.sqrt(np.mean((mLL - mRR)[band] ** 2)), m['gainA']))
print(time.time() - t0)
