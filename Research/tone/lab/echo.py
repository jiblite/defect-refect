# Echo finder, lab version: a short reflection (comb filter) shows as a peak in the cepstrum of the long-term log spectrum, at the
# reflection's delay, with the reflection's gain as its height. A pitched note does the same while it sounds, so the peak has to be
# there in every part of the sound (three thirds in time, quiet half, loud half).
import numpy as np
import feat
SR = 48000; N = feat.N; FB = feat.FB; QM = 1441                 # quefrencies up to 30 ms
_band = (FB >= 100) & (FB <= 16000); W0 = np.zeros_like(FB); W0[_band] = np.hanning(_band.sum())
def groups(Q):
    """Q: power (T, bins), frames that count. Mean clamped log spectrum (dB) of: all frames, three thirds in time, the quiet and the loud half."""
    top = np.percentile(Q, 95, axis=0); L = 10 * np.log10(np.maximum(Q, np.maximum(top * 1e-4, 1e-30))); T = len(L); e = Q.sum(1); o = np.argsort(e); t = np.arange(T)
    sel = [np.ones(T, bool), t < T / 3, (t >= T / 3) & (t < 2 * T / 3), t >= 2 * T / 3, np.isin(t, o[:T // 2]), np.isin(t, o[T // 2:])]
    lev = 10 * np.log10(np.maximum(top, 1e-30)); return np.stack([L[s].mean(0) if s.any() else L.mean(0) for s in sel]), lev
def weight(lev):
    """Bins that count for the cepstrum: 100 Hz..16 kHz, tapered, and only where the sound has something (within 60 dB of its loudest region)."""
    k = np.ones(33) / 33; sm = np.convolve(lev, k, 'same'); v = np.clip((sm - (sm[_band].max() - 60.0)) / 10.0, 0, 1); return W0 * v
def ceps(Ls, w):
    """Ls: (K, bins) dB. Cepstrum (K, QM) scaled so that a reflection of gain g at delay d samples gives g at index d."""
    L = Ls * (np.log(10) / 20); L = (L - (L * w).sum(-1, keepdims=True) / w.sum()) * w
    return np.fft.irfft(L, N, axis=-1)[..., :QM] * (N / w.sum())
def steady(c):
    """(K, QM) -> the smallest magnitude over the parts where all parts agree in sign, else 0."""
    same = (np.sign(c) == np.sign(c[0])).all(0); return np.where(same, np.sign(c[0]) * np.abs(c).min(0), 0.0)
def analyse(Q):
    Ls, lev = groups(Q); w = weight(lev); c = ceps(Ls, w); return c, w, Ls[0]
