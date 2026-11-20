# The fixed filter's family and how it is applied: a magnitude curve on a frequency grid, used as a minimum-phase filter.
import numpy as np, scipy.signal as ss
SR = 48000
def biquad_db(f, kind, fc, q):
    """RBJ low-pass, high-pass or band-pass (peak gain 1) at fc with Q: magnitude in dB at the frequencies f (analog prototype warped as the digital one is)."""
    w0 = 2 * np.pi * fc / SR; al = np.sin(w0) / (2 * q); c = np.cos(w0)
    if kind == 'low': b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2])
    elif kind == 'high': b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    else: b = np.array([al, 0, -al])
    a = np.array([1 + al, -2 * c, 1 - al]); w, h = ss.freqz(b, a, 2 * np.pi * np.asarray(f) / SR); return 20 * np.log10(np.abs(h) + 1e-12)
def curve_db(f, kind, fc, q, steep):
    """The family: one section with the resonance, and for the steep kind a second one without (Q 0.707)."""
    d = biquad_db(f, kind, fc, q)
    return d + biquad_db(f, kind, fc, 0.7071) if steep else d
def minphase(curve_db_fn, n=16384):
    """A minimum-phase impulse response of n taps with the magnitude curve_db_fn(f) (dB)."""
    nfft = 4 * n; f = np.fft.rfftfreq(nfft, 1 / SR); L = np.asarray(curve_db_fn(np.maximum(f, 1.0))) * np.log(10) / 20
    cep = np.fft.irfft(L); cep[1:nfft // 2] *= 2; cep[nfft // 2 + 1:] = 0; h = np.fft.irfft(np.exp(np.fft.rfft(cep)))[:n]
    k = n // 8; h[-k:] *= 0.5 * (1 + np.cos(np.pi * (np.arange(k) + 0.5) / k)); return h
def apply(x, curve_db_fn):
    h = minphase(curve_db_fn); return np.stack([ss.fftconvolve(x[:, c], h)[:len(x)] for c in range(x.shape[1])], 1)
def softcap(g, boost, cut):
    def one(v, cap): k = 0.75 * cap; return np.where(v <= k, v, k + (cap - k) * np.tanh((v - k) / max(cap - k, 1e-9)))
    g = np.asarray(g, float); return np.where(g >= 0, one(g, boost), -one(-g, cut))
