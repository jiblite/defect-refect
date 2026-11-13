# Long-term spectrum statistics, lab version: several ways to boil the frames down to one dB value per bin.
import numpy as np, scipy.signal as ss
SR = 48000
_w = {}
def frames_db(x, N=8192, hop=None):
    """(n,2) -> power per frame and bin, (2, T, N/2+1), Hann window."""
    hop = hop or N // 4
    if N not in _w: _w[N] = ss.get_window('hann', N, fftbins=True)
    xp = np.vstack([np.zeros((N // 2, 2)), x, np.zeros((N, 2))]); T = 1 + (len(xp) - N) // hop
    idx = np.arange(N)[None, :] + hop * np.arange(T)[:, None]
    return np.stack([np.abs(np.fft.rfft(xp[idx, c] * _w[N], axis=1)) ** 2 for c in range(2)])
def active(P, gate_db=45.0):
    """Frames that count: total power within gate_db of the 95th-percentile frame."""
    e = P.sum((0, 2)); ref = np.percentile(e, 95); return e > ref * 10 ** (-gate_db / 10)
def stats(P, clamp_db=40.0, gate_db=45.0, link=True):
    """Dict of per-bin dB statistics over the active frames. link: both channels pooled into one curve (power summed), else per channel."""
    a = active(P, gate_db); Q = P[:, a, :]
    if link: Q = Q.sum(0, keepdims=True)
    out = {}
    out['pmean'] = 10 * np.log10(Q.mean(1) + 1e-30)
    top = np.percentile(Q, 95, axis=1); fl = np.maximum(top * 10 ** (-clamp_db / 10), 1e-30)       # each bin's own floor: clamp_db under its own loud frames
    L = 10 * np.log10(np.maximum(Q, fl[:, None, :]))
    out['lmean'] = L.mean(1)
    for q in (25, 50, 75, 90): out['p%d' % q] = np.percentile(L, q, axis=1)
    out['top'] = 10 * np.log10(top + 1e-30)
    return out
def to_grid(f, v, grid):
    """Mean of v over each log-grid cell (cells narrower than two bins: interpolated)."""
    g = np.asarray(grid, float); e = np.empty(len(g) + 1); e[1:-1] = np.sqrt(g[:-1] * g[1:]); e[0] = g[0] ** 2 / e[1]; e[-1] = g[-1] ** 2 / e[-2]
    df = f[1] - f[0]; cs = np.concatenate([[0.0], np.cumsum(v)]); lo = np.clip(np.ceil(e[:-1] / df).astype(int), 0, len(f)); hi = np.clip(np.ceil(e[1:] / df).astype(int), 0, len(f))
    out = np.interp(g, f, v); ok = hi - lo >= 2; out[ok] = (cs[hi[ok]] - cs[lo[ok]]) / (hi[ok] - lo[ok]); return out
def gsmooth(v, sigma):
    """Gaussian smoothing along the last axis with reflected ends; sigma in grid points."""
    if sigma <= 0: return v
    k = int(np.ceil(4 * sigma)); t = np.arange(-k, k + 1); w = np.exp(-0.5 * (t / sigma) ** 2); w /= w.sum()
    vp = np.concatenate([v[..., k:0:-1], v, v[..., -2:-k - 2:-1]], -1)
    return np.apply_along_axis(lambda a: np.convolve(a, w, 'valid'), -1, vp)
