# The few signal tools the chorus engine stands on, written out step by step so that the app's code can do exactly the same.
#   dec2(y)          the sound at half the sample rate (41 taps, Kaiser 5, as scipy's resample_poly(y, 1, 2))
#   bands(y)         the sound in NB frequency ranges that add up to the sound again (masks on a 4096 transform, hop 1024)
#   spec(x)          a plain picture for measuring: frames of 2048 every 1024, Hann
#   smooth2(P, a, b) running mean over a bins and b frames, the edges held
import numpy as np
SR = 48000
EDGES = np.array([150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 9600.0]); NB = len(EDGES) + 1
BN = 4096; BH = 1024                                        # the transform the ranges are cut with
SN = 2048; SH = 1024                                        # the transform the measuring is done on

def _i0(x):
    s = 1.0; t = 1.0
    for k in range(1, 40): t *= (x / (2.0 * k)) ** 2; s += t
    return s

def _taps():
    m = np.arange(41) - 20.0; w = np.array([_i0(5.0 * np.sqrt(max(0.0, 1.0 - (v / 20.0) ** 2))) for v in m]) / _i0(5.0); h = 0.5 * np.sinc(0.5 * m) * w; return h / h.sum()
TAPS = _taps()

def dec2(y):
    n = len(y); m = (n + 1) // 2; yp = np.concatenate([np.zeros(20), y, np.zeros(22)]); out = np.zeros(m)
    for j in range(41): out += TAPS[j] * yp[20 + 2 * np.arange(m) + 20 - j]
    return out

def _masks():
    f = np.arange(BN // 2 + 1) * SR / BN; lp = []
    for e in EDGES:
        u = np.clip(np.log2(np.maximum(f, 1e-9) / e) / 0.25, -1.0, 1.0); lp.append(0.5 - 0.5 * np.sin(0.5 * np.pi * u))        # 1 a quarter octave under the edge, 0 a quarter octave over it
    lp = np.array(lp); M = np.zeros((NB, len(f))); M[0] = lp[0]
    for b in range(1, NB - 1): M[b] = lp[b] - lp[b - 1]
    M[NB - 1] = 1.0 - lp[-1]; return M
MASKS = _masks()

def bands(y):
    n = len(y); J = n // BH + 1 + BN // BH; win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(BN) / BN)); out = np.zeros((NB, n)); nrm = np.zeros(n); yp = np.concatenate([np.zeros(BN), y, np.zeros(BN + BH)])
    for j in range(J):
        s = j * BH - BN; seg = yp[s + BN:s + 2 * BN] * win; lo = max(s, 0); hi = min(s + BN, n)
        if hi <= lo: continue
        S = np.fft.rfft(seg); nrm[lo:hi] += (win * win)[lo - s:hi - s]
        if not np.any(seg): continue
        for b in range(NB): out[b, lo:hi] += (np.fft.irfft(S * MASKS[b], BN) * win)[lo - s:hi - s]
    return out / np.maximum(nrm, 1e-9)[None, :]

def spec(x):
    n = len(x); T = max((n - SN) // SH + 1, 0); win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(SN) / SN)
    if T <= 0: return np.zeros((SN // 2 + 1, 0), complex)
    return np.fft.rfft(x[np.arange(T)[:, None] * SH + np.arange(SN)[None, :]] * win, axis=1).T        # (bins, frames)

def _run(P, k, axis):
    """Running mean over k (odd) along an axis, the edge values held outside."""
    h = k // 2; Q = np.moveaxis(P, axis, 0); pad = np.concatenate([np.repeat(Q[:1], h, 0), Q, np.repeat(Q[-1:], h, 0)]); c = np.cumsum(np.concatenate([np.zeros((1,) + Q.shape[1:]), pad]), 0)
    return np.moveaxis((c[k:] - c[:-k]) / k, 0, axis)

def smooth2(P, a, b): return _run(_run(P, a, 0), b, 1)

if __name__ == '__main__':
    import scipy.signal as ss
    r = np.random.default_rng(0); y = r.standard_normal(100001); a = dec2(y); b = ss.resample_poly(y, 1, 2); print('dec2 against resample_poly: lengths', len(a), len(b), 'largest difference %.2e' % np.abs(a - b).max())
    B = bands(y); print('ranges add up to the sound within %.2e' % np.abs(B.sum(0) - y).max()); f, Pw = ss.welch(B, SR, nperseg=8192, axis=1)
    for i in range(NB): print('   range %d: 90%% of its power between %.0f and %.0f Hz' % (i, f[np.searchsorted(np.cumsum(Pw[i]) / Pw[i].sum(), 0.05)], f[np.searchsorted(np.cumsum(Pw[i]) / Pw[i].sum(), 0.95)]))
    import scipy.ndimage as ndi
    P = r.random((40, 30)); print('smooth2 against uniform_filter: %.2e' % np.abs(smooth2(P, 9, 5) - ndi.uniform_filter(P, size=(9, 5), mode='nearest')).max())
    S1 = spec(y); S2 = ss.stft(y, SR, 'hann', SN, SN - SH, boundary=None, padded=False)[2] * (0.5 * SN); print('spec against scipy stft: %.2e' % np.abs(S1 - S2).max(), S1.shape, S2.shape)
