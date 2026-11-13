# What the tone model sees of a sound: long-term statistics on a 1/24-octave grid. Every statistic is "the curve plus something of the
# source": a fixed filter with gain C(f) dB moves the first one (lmean) up by exactly C(f) and leaves the others (differences) alone.
import numpy as np, scipy.signal as ss, scipy.ndimage as ndi
SR = 48000; N = 8192; HOP = 2048; PER_OCT = 24
GRID = 20.0 * 2.0 ** (np.arange(int(np.floor(np.log2(1000.0) * PER_OCT + 1e-9)) + 1) / PER_OCT)       # 20 Hz .. 20 kHz
G = len(GRID); FB = np.fft.rfftfreq(N, 1.0 / SR)
_win = ss.get_window('hann', N, fftbins=True)
NAMES = ['lmean', 'floor', 'env', 'p10', 'p25', 'p50', 'p75', 'p90', 'top', 'pmean', 'flux', 'onset', 'tonal']
FW, FH, EW = 16, 8, 25          # floor: median of 16-bin windows every 8 bins; env: maximum over 25 bins (both per frame, then averaged over frames)
FC = (np.arange((N // 2 + 1 - FW) // FH + 1) * FH + (FW - 1) / 2.0) * SR / N       # centre frequencies of the floor windows
def frames(x):
    """(n,2) -> power (2, T, N/2+1)."""
    xp = np.vstack([np.zeros((N // 2, 2)), x, np.zeros((N, 2))]); T = 1 + (len(xp) - N) // HOP
    idx = np.arange(N)[None, :] + HOP * np.arange(T)[:, None]
    return np.stack([np.abs(np.fft.rfft(xp[idx, c] * _win, axis=1)) ** 2 for c in range(2)])
def active(P, gate_db=45.0):
    e = P.sum((0, 2)); return e > np.percentile(e, 95) * 10 ** (-gate_db / 10)
_e = np.empty(G + 1); _e[1:-1] = np.sqrt(GRID[:-1] * GRID[1:]); _e[0] = GRID[0] ** 2 / _e[1]; _e[-1] = GRID[-1] ** 2 / _e[-2]
_lo = np.clip(np.ceil(_e[:-1] / (SR / N)).astype(int), 0, len(FB)); _hi = np.clip(np.ceil(_e[1:] / (SR / N)).astype(int), 0, len(FB)); _wide = (_hi - _lo) >= 2
def to_grid(v):
    """Mean of a per-bin curve over each grid cell; cells narrower than two bins take the straight line between bins."""
    v = np.asarray(v, float); cs = np.concatenate([[0.0], np.cumsum(v)]); out = np.interp(GRID, FB, v)
    out[_wide] = (cs[_hi[_wide]] - cs[_lo[_wide]]) / (_hi[_wide] - _lo[_wide]); return out
def bin_stats(Q, clamp_db=40.0):
    """Q: power (T, bins) of the frames that count. Returns dict of per-bin curves in dB (see NAMES)."""
    top = np.percentile(Q, 95, axis=0); fl = np.maximum(top * 10 ** (-clamp_db / 10), 1e-30)
    L = 10 * np.log10(np.maximum(Q, fl[None, :])); o = {}
    o['lmean'] = L.mean(0); pc = np.percentile(L, [10, 25, 50, 75, 90], axis=0)
    for k, q in enumerate((10, 25, 50, 75, 90)): o['p%d' % q] = pc[k]
    o['top'] = 10 * np.log10(np.maximum(top, 1e-30)); o['pmean'] = 10 * np.log10(np.maximum(Q.mean(0), 1e-30))
    dL = np.diff(L, axis=0); o['flux'] = np.abs(dL).mean(0) if len(dL) else np.zeros(L.shape[1])
    if len(dL):
        on = np.maximum(dL, 0).sum(1); k = max(1, int(round(0.2 * len(on)))); sel = np.argsort(on)[-k:] + 1; o['onset'] = L[sel].mean(0)
    else: o['onset'] = o['lmean'].copy()
    o['floor'] = np.interp(FB, FC, np.median(np.lib.stride_tricks.sliding_window_view(L, FW, axis=1)[:, ::FH], axis=-1).mean(0))
    o['env'] = ndi.maximum_filter1d(L, EW, axis=1, mode='nearest').mean(0)
    o['tonal'] = np.abs(L - ndi.uniform_filter1d(L, 9, axis=1, mode='nearest')).mean(0)
    return o
def grid_feats(Q):
    """(len(NAMES), G): lmean as it is (dB); floor, env, the percentiles, top, pmean and onset as differences to lmean; flux and tonal as they are."""
    o = bin_stats(Q); F = np.stack([to_grid(o[k]) for k in NAMES])
    for k in ('floor', 'env', 'p10', 'p25', 'p50', 'p75', 'p90', 'top', 'pmean', 'onset'): F[NAMES.index(k)] -= F[0]
    return F
def feats(x):
    """(n,2) -> (3, K, G): left, right, both channels pooled."""
    P = frames(x); a = active(P); P = P[:, a, :]
    return np.stack([grid_feats(P[0]), grid_feats(P[1]), grid_feats(P[0] + P[1])]).astype(np.float32)
