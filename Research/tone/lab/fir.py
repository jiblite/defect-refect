# Curve (dB on any frequency grid) -> FIR, and applying it. First lab version; the final one lives in tone.py.
import numpy as np, scipy.signal as ss
SR = 48000
def log_grid(per_oct=24, fmin=20.0, fmax=20000.0):
    n = int(np.floor(np.log2(fmax / fmin) * per_oct + 1e-9)) + 1; return fmin * 2.0 ** (np.arange(n) / per_oct)
def lin_grid(nfft): return np.fft.rfftfreq(nfft, 1.0 / SR)
def bins_db(grid_hz, curve_db, nfft):
    """The curve on the FFT bins of an nfft-point transform: straight lines between grid points on a log-frequency axis, end values held."""
    f = np.fft.rfftfreq(nfft, 1.0 / SR); lg = np.log(np.maximum(grid_hz, 1e-3)); lf = np.log(np.maximum(f, 1e-3)); return np.interp(lf, lg, curve_db)
def minphase(mag_db, nfft):
    L = mag_db * (np.log(10.0) / 20.0); c = np.fft.irfft(L, nfft); w = np.zeros(nfft); w[0] = 1.0; w[1:nfft // 2] = 2.0; w[nfft // 2] = 1.0
    return np.fft.irfft(np.exp(np.fft.rfft(c * w)), nfft)
def to_ir(grid_hz, curve_db, n=8192, phase='minimum', over=4):
    """FIR of n taps with the curve as its magnitude response. minimum: starts at tap 0. linear: symmetric around tap n//2."""
    curve_db = np.asarray(curve_db, float)
    if curve_db.ndim == 2: return np.stack([to_ir(grid_hz, c, n, phase, over) for c in curve_db], 1)
    nfft = int(2 ** np.ceil(np.log2(n * over))); db = bins_db(grid_hz, curve_db, nfft)
    if phase == 'minimum':
        h = minphase(db, nfft)[:n].copy(); k = max(8, n // 8); h[-k:] *= 0.5 * (1 + np.cos(np.pi * np.arange(k) / k)); return h
    h = np.fft.irfft(10.0 ** (db / 20.0), nfft); h = np.roll(h, n // 2)[:n].copy(); k = n // 4
    w = np.ones(n); r = 0.5 * (1 - np.cos(np.pi * (np.arange(k) + 0.5) / k)); w[:k] = r; w[-k:] = r[::-1]; return h * w
def apply_ir(x, h, phase='minimum'):
    """x (n,2); h (taps,) or (taps,2). Same length out; the linear-phase delay is taken out."""
    if h.ndim == 1: h = np.stack([h, h], 1)
    n = len(x); d = 0 if phase == 'minimum' else h.shape[0] // 2
    return np.stack([ss.fftconvolve(x[:, c], h[:, c])[d:d + n] for c in range(2)], 1)
