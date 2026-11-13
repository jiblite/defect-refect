"""TONE engine prototype (V2). 48 kHz, stereo (n, 2) float arrays. Needs numpy and scipy; the optional model also tone_model.npz.

What it does
  contour(x)                      the tone curve found on one sound, blind: (grid_hz, curve_db)
  smooth(grid_hz, curve_db, d)    the Detail control (0 = broad strokes only .. 1 = everything)
  apply_curve(x, grid, curve, a)  fader on the analysed sound: a = 0 curve removed, 1 untouched, 2 applied once more
  put_curve(y, grid, curve, a)    the found curve onto a second sound: a = 0 none, 1 as found, 2 double
  match(a, b, detail)             Tone Match: the curve that gives b the long-term spectrum of a (apply with put_curve)
  to_ir(grid, curve, n, phase)    the curve as an impulse response, 'minimum' or 'linear' phase

How the blind curve is found
  1. Echo finder (always on): a short reflection (comb filter) is a peak in the cepstrum of the long-term log spectrum, at the
     reflection's delay and with its gain as height. It only counts when it passes four tests (see find_echoes), among them that every
     sixth of the sound shows it, which a played note does not. Result: delay and gain, hence an exact comb curve. On a sound
     without a reflection it finds nothing and the curve is exactly zero: the fader then does nothing at all.
  2. Model (off unless asked for: contour(x, model=True)): a small convolution net (73 042 weights) along a 1/24-octave grid. It sees
     long-term statistics of the sound (_bin_stats, _net_input) and gives the curve in dB per cell plus its own doubt per cell. It was
     trained on clean clips plus random curves, so it knows what long-term spectra of clean sounds tend to look like. Its curve is
     shrunk where it is small against its doubt (restraint). It is a guess, not a measurement: on equalised sounds it recovers a small
     part of the curve, and on some clean sounds it invents one (numbers in REPORT.md). That is why it is not on by default.
The curve lives on the analysis bins (4097 points, 5.86 Hz apart, 0..24 kHz): fine enough for comb teeth at any frequency; a
log-frequency grid of 1/24 octave is not (see REPORT.md). Any other increasing grid works with the functions below as well.

Numbers that matter for a port: frames N = 8192, hop 2048, Hann window, frames within 45 dB of the loud ones; model grid 240 cells
from 20 Hz to 20 kHz; FIR default 16384 taps, minimum phase (FFT size 4 x taps for the design); gain caps +18 dB (boost) and -30 dB
(cut), applied after the fader, with a soft knee from 3/4 of the cap.
"""
import os
import numpy as np, scipy.signal as ss, scipy.ndimage as ndi

SR = 48000
N, HOP = 8192, 2048
PER_OCT = 24
GRID = 20.0 * 2.0 ** (np.arange(int(np.floor(np.log2(1000.0) * PER_OCT + 1e-9)) + 1) / PER_OCT)      # model grid: 240 cells, 20 Hz .. 20 kHz
G = len(GRID)
FB = np.fft.rfftfreq(N, 1.0 / SR)                                                                    # the curve's own grid: the analysis bins
MAX_BOOST_DB, MAX_CUT_DB = 18.0, 30.0
MODEL_MIN_DB = 1.0                 # a model curve that nowhere reaches this many dB counts as 'nothing found'
_HERE = os.path.dirname(os.path.abspath(__file__))
_win = ss.get_window('hann', N, fftbins=True)

# ------------------------------------------------------------------------------------------------ analysis: frames and statistics
def _stereo(x):
    """Any input as float64 (n, 2): a mono sound is put on both channels."""
    x = np.asarray(x, np.float64); return np.stack([x, x], 1) if x.ndim == 1 else x

def _frames(x):
    """(n, 2) -> power per frame and bin, (2, T, N/2+1). Half a window of silence in front, a window behind."""
    x = _stereo(x)
    xp = np.vstack([np.zeros((N // 2, 2)), x, np.zeros((N, 2))]); T = 1 + (len(xp) - N) // HOP
    idx = np.arange(N)[None, :] + HOP * np.arange(T)[:, None]
    return np.stack([np.abs(np.fft.rfft(xp[idx, c] * _win, axis=1)) ** 2 for c in range(2)])

def _active(P, gate_db=45.0):
    """Frames that count: total power within gate_db of the loud frames (95th percentile). Silence and faded tails are left out."""
    e = P.sum((0, 2)); return e > np.percentile(e, 95) * 10 ** (-gate_db / 10)

_e = np.empty(G + 1); _e[1:-1] = np.sqrt(GRID[:-1] * GRID[1:]); _e[0] = GRID[0] ** 2 / _e[1]; _e[-1] = GRID[-1] ** 2 / _e[-2]          # cell edges
_lo = np.clip(np.ceil(_e[:-1] / (SR / N)).astype(int), 0, len(FB)); _hi = np.clip(np.ceil(_e[1:] / (SR / N)).astype(int), 0, len(FB)); _wide = (_hi - _lo) >= 2
def _to_grid(v):
    """Per-bin curve -> mean over each grid cell; cells narrower than two bins take the straight line between bins."""
    v = np.asarray(v, float); cs = np.concatenate([[0.0], np.cumsum(v)]); out = np.interp(GRID, FB, v)
    out[_wide] = (cs[_hi[_wide]] - cs[_lo[_wide]]) / (_hi[_wide] - _lo[_wide]); return out

def _to_bins(grid_hz, curve_db, f=FB):
    """A curve given on any increasing grid -> on the frequencies f: straight lines on a log-frequency axis, end values held."""
    return np.interp(np.log(np.maximum(f, 1e-3)), np.log(np.maximum(np.asarray(grid_hz, float), 1e-3)), curve_db)

NAMES = ['lmean', 'floor', 'env', 'p10', 'p25', 'p50', 'p75', 'p90', 'top', 'pmean', 'flux', 'onset', 'tonal']
_FW, _FH, _EW = 16, 8, 25
_FC = (np.arange((N // 2 + 1 - _FW) // _FH + 1) * _FH + (_FW - 1) / 2.0) * SR / N
def _clamped_log(Q, clamp_db=40.0):
    """dB per frame and bin; every bin is held at clamp_db under its own loud frames (95th percentile over time). A fixed filter moves
    all of a bin's values, floor included, by the same amount: every statistic below shifts by exactly the filter's gain."""
    top = np.percentile(Q, 95, axis=0); return 10 * np.log10(np.maximum(Q, np.maximum(top * 10 ** (-clamp_db / 10), 1e-30))), top

def _bin_stats(Q):
    """Q: power (T, bins) of the frames that count. Long-term statistics per bin, in dB:
    lmean  mean over frames of the clamped log power (the main carrier of the curve)
    floor  per frame the median of 16-bin windows (94 Hz, every 8 bins), then the mean over frames: what lies between the partials
    env    per frame the maximum over 25 bins, mean over frames (kept for reference, not used by the model)
    p10..p90  percentiles over frames;  top  the 95th percentile of the power;  pmean  plain power mean
    flux   mean absolute change from frame to frame;  onset  mean over the fifth of the frames with the strongest rise
    tonal  mean absolute difference to the 9-bin neighbourhood, per frame (how line-like the spectrum is)"""
    L, top = _clamped_log(Q); o = {}
    o['lmean'] = L.mean(0); pc = np.percentile(L, [10, 25, 50, 75, 90], axis=0)
    for k, q in enumerate((10, 25, 50, 75, 90)): o['p%d' % q] = pc[k]
    o['top'] = 10 * np.log10(np.maximum(top, 1e-30)); o['pmean'] = 10 * np.log10(np.maximum(Q.mean(0), 1e-30))
    dL = np.diff(L, axis=0); o['flux'] = np.abs(dL).mean(0) if len(dL) else np.zeros(L.shape[1])
    if len(dL):
        on = np.maximum(dL, 0).sum(1); k = max(1, int(round(0.2 * len(on)))); sel = np.argsort(on)[-k:] + 1; o['onset'] = L[sel].mean(0)
    else: o['onset'] = o['lmean'].copy()
    o['floor'] = np.interp(FB, _FC, np.median(np.lib.stride_tricks.sliding_window_view(L, _FW, axis=1)[:, ::_FH], axis=-1).mean(0))
    o['env'] = ndi.maximum_filter1d(L, _EW, axis=1, mode='nearest').mean(0)
    o['tonal'] = np.abs(L - ndi.uniform_filter1d(L, 9, axis=1, mode='nearest')).mean(0)
    return o

def _grid_feats(o):
    """(13, G): the statistics on the grid; lmean as it is, the level statistics as differences to lmean, flux and tonal as they are."""
    F = np.stack([_to_grid(o[k]) for k in NAMES])
    for k in ('floor', 'env', 'p10', 'p25', 'p50', 'p75', 'p90', 'top', 'pmean', 'onset'): F[NAMES.index(k)] -= F[0]
    return F

# ------------------------------------------------------------------------------------------------ part 2: the model
_USE = [NAMES.index(k) for k in ('floor', 'p10', 'p25', 'p50', 'p75', 'p90', 'top', 'pmean', 'flux', 'onset', 'tonal')]
_SC = np.array([10, 10, 10, 10, 10, 10, 10, 10, 5, 10, 5], np.float32)
_CIN = len(_USE) + 3
_POS = np.linspace(-1, 1, G).astype(np.float32)
_t = np.arange(-48, 49); _gk = np.exp(-0.5 * (_t / 12.0) ** 2); _gk /= _gk.sum()
def _trend(v):
    """Gaussian trend along the grid, sigma 12 cells (half an octave), ends reflected."""
    return np.convolve(np.concatenate([v[48:0:-1], v, v[-2:-50:-1]]), _gk, 'valid')

def _net_input(F):
    """(13, G) statistics -> (14, G) network input: 0 level under the loudest region / 20 dB; 1 lmean minus its half-octave trend / 10 dB;
    2..12 floor, p10, p25, p50, p75, p90, top, pmean, flux, onset, tonal, each divided by its scale (10 dB; flux and tonal 5 dB);
    13 position on the axis (-1 at 20 Hz .. +1 at 20 kHz). Everything clipped to a few units."""
    F = np.asarray(F, np.float32); lm = F[0]; tr = _trend(lm).astype(np.float32); x = np.empty((_CIN, G), np.float32)
    x[0] = np.clip((lm - tr.max()) / 20.0, -4.0, 1.0); x[1] = np.clip((lm - tr) / 10.0, -3.0, 3.0); x[2:_CIN - 1] = np.clip(F[_USE] / _SC[:, None], -4.0, 4.0); x[_CIN - 1] = _POS
    return x

_model = None
def _load_model():
    global _model
    if _model is None:
        z = np.load(os.path.join(_HERE, 'tone_model.npz')); _model = {k: z[k] for k in z.files}
    return _model

def _net(x):
    """The net in plain numpy: a 1x1 input layer, residual blocks h += conv5(relu(h)) with dilations (zeros outside the grid), a 1x1 output
    layer on relu(h). Two outputs per cell: the curve (times 10 = dB) and the net's own doubt b = exp(second output, held to -3..3), in dB:
    the absolute error it expects in that cell."""
    M = _load_model(); h = M['inp_w'] @ x + M['inp_b'][:, None]
    for i, d in enumerate(M['dil']):
        a = np.maximum(h, 0.0); w = M['c%d_w' % i]; acc = np.repeat(M['c%d_b' % i][:, None], G, 1)
        for j in range(w.shape[2]):
            s = (j - w.shape[2] // 2) * int(d)
            if s == 0: acc += w[:, :, j] @ a
            elif s > 0: acc[:, :G - s] += w[:, :, j] @ a[:, s:]
            else: acc[:, -s:] += w[:, :, j] @ a[:, :G + s]
        h = h + acc
    o = (M['out_w'] @ np.maximum(h, 0.0) + M['out_b'][:, None]).astype(np.float64)
    return o[0] * 10.0, (np.exp(np.clip(o[1], -3.0, 3.0)) if len(o) > 1 else np.ones(G))

def _valid(F):
    """Which grid cells carry sound (0..1): the loud frames' level (lmean + top) within 60..70 dB of its largest."""
    lev = F[0] + F[NAMES.index('top')]; return np.clip((lev - (lev.max() - 70.0)) / 10.0, 0.0, 1.0)

def _wmedian(v, w):
    o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, 0.5 * c[-1])]) if c[-1] > 0 else 0.0

def _shrink(c, theta):
    """Restraint: values well under theta dB vanish, values well above stay: c * c^2 / (c^2 + theta^2). theta may differ per cell."""
    theta = np.asarray(theta, float); return c * c * c / (c * c + theta * theta + 1e-12)

# ------------------------------------------------------------------------------------------------ part 1: the echo finder
_QM = 1441                                                                         # delays up to 30 ms are computed, 0.5..15 ms are searched
_QLO, _QHI = int(0.5 * 48), int(15 * 48)
_band = (FB >= 100) & (FB <= 16000); _W0 = np.zeros_like(FB); _W0[_band] = np.hanning(_band.sum())
ECHO_MIN_GAIN, ECHO_MIN_GAIN_SHORT = 0.25, 0.40      # smallest reflection gain that counts: delays of 1 ms and more, delays under 1 ms

def _lmean_ln(Q, clamp_db=40.0):
    """ln-magnitude long-term spectrum (mean over frames of the log power, each bin held at clamp_db under its own loud frames) and the
    loud frames' level in dB. A fixed filter adds its own ln-magnitude to the first, whatever the sound does."""
    top = np.percentile(Q, 95, axis=0); return 0.5 * np.log(np.maximum(Q, np.maximum(top * 10 ** (-clamp_db / 10), 1e-30))).mean(0), 10 * np.log10(np.maximum(top, 1e-30))

def _echo_weight(lev):
    """Bins that count for the cepstrum: 100 Hz .. 16 kHz with a Hann taper, and only where the sound has something. The level is smoothed
    over 1.5 kHz first: otherwise a comb's own notches print themselves into the weights and smear its cepstral peak into side peaks."""
    sm = ndi.uniform_filter1d(np.maximum(lev, lev[_band].max() - 80.0), 257, mode='nearest'); return _W0 * np.clip((sm - (sm[_band].max() - 60.0)) / 10.0, 0, 1)

def _ceps(Lm, w):
    """Cepstrum of the weighted ln-magnitude spectrum, scaled so that a reflection of gain g at a delay of d samples gives g at index d."""
    x = (Lm - (Lm * w).sum() / max(w.sum(), 1e-9)) * w; return np.fft.irfft(x, N)[:_QM] * (N / max(w.sum(), 1e-9))

def _project(Lm, w, d):
    """Gain of the ripple cos(2 pi f d) in the weighted ln-magnitude spectrum, for (fractional) delays d in samples."""
    d = np.atleast_1d(d).astype(float); x = (Lm - (Lm * w).sum() / max(w.sum(), 1e-9)) * w
    return (2.0 / max(w.sum(), 1e-9)) * (np.cos(2 * np.pi * np.outer(d, np.arange(len(Lm))) / N) @ x)

def comb_db(delay, gain, f=FB):
    """20 log10 |1 + gain * e^{-j 2 pi f delay / SR}| : the curve of one reflection (delay in samples)."""
    return 10 * np.log10(np.maximum(1 + gain * gain + 2 * gain * np.cos(2 * np.pi * np.asarray(f, float) * delay / SR), 1e-12))

def _echo_refine(Lm, w, d, g, span=1.0, step=0.02, K=4):
    """Delay and gain by a least-squares fit of the whole comb curve ln|1 + g e^{-j theta}| = sum_k (-1)^(k+1) g^k / k cos(k theta), first K
    terms, over delays d +- span and gains 0.7 g .. 1.3 g."""
    dd = d + np.arange(-span, span + 1e-9, step); P = np.stack([_project(Lm, w, k * dd) for k in range(1, K + 1)]); best = None
    for gg in g * np.arange(0.7, 1.31, 0.05):
        co = np.array([(-1.0) ** (k + 1) * gg ** k / k for k in range(1, K + 1)]); sc = -2 * (co[:, None] * P).sum(0) + (co ** 2).sum(); j = int(np.argmin(sc))
        if best is None or sc[j] < best[0]: best = (float(sc[j]), float(dd[j]), float(gg))
    return best[1], best[2]

def find_echoes(Q, max_echoes=2, sens=1.0, ncand=40):
    """Q: power (T, bins) of the frames that count (one channel, or both pooled). Returns [(delay in samples, gain)], strongest first.
    A candidate is a peak (0.5..15 ms) in the cepstrum of the whole sound's long-term log spectrum. It is taken as a reflection when
      1. its gain is at least ECHO_MIN_GAIN (ECHO_MIN_GAIN_SHORT for delays under 1 ms, where a sound's own formants ripple too), over sens;
      2. its second peak, at twice the delay, is negative: at most -0.1 g^2 (a reflection has -0.5 g^2, a steady pitch a positive one);
      3. every sixth of the sound (in time, each with its own long-term spectrum) shows it with the same sign and at least 0.4 of its size
         (sixths with less than half of the whole sound's frequency range are left out; three must remain);
      4. it stands alone: after delay and gain are refined (_echo_refine) and its comb is taken out, nothing of half its size is left in
         the cepstrum within +-10 samples of its delay (a cluster of peaks is a ripple of the sound itself).
    The comb is then taken out of the spectra and the next reflection is looked for."""
    T = len(Q); nseg = min(6, T // 8)
    if nseg < 3: return []
    parts = [Q] + [Q[i * T // nseg:(i + 1) * T // nseg] for i in range(nseg)]; LM = []; WW = []
    for p in parts: Lm, lev = _lmean_ln(p); LM.append(Lm); WW.append(_echo_weight(lev))
    if WW[0].sum() < 50: return []
    use = [i for i in range(1, nseg + 1) if WW[i].sum() >= 0.5 * WW[0].sum()]
    if len(use) < 3: return []
    thr = lambda d: (ECHO_MIN_GAIN_SHORT if d < 48 else ECHO_MIN_GAIN) / sens; tq = np.where(np.arange(_QM) < 48, ECHO_MIN_GAIN_SHORT, ECHO_MIN_GAIN) / sens; out = []
    for _ in range(max_echoes):
        a = np.abs(_ceps(LM[0], WW[0])) / tq; a[:_QLO] = 0; a[_QHI:] = 0; cands = []
        for _c in range(ncand):
            k = int(np.argmax(a))
            if a[k] < 0.8: break
            a[max(0, k - 3):k + 4] = 0
            dd = k + np.arange(-1.5, 1.5001, 0.05); pr = _project(LM[0], WW[0], dd); j = int(np.argmax(np.abs(pr))); d = float(dd[j]); g = float(pr[j])
            if abs(g) < thr(d): continue                                                                   # 1
            if float(_project(LM[0], WW[0], 2 * d)[0]) > -0.1 * g * g: continue                              # 2
            gs = [float(_project(LM[i], WW[i], d)[0]) for i in use]
            if not all(np.sign(v) == np.sign(g) and abs(v) >= 0.4 * abs(g) for v in gs): continue            # 3
            cands.append((abs(g) / thr(d), d, float(np.median([g] + gs))))
        got = None
        for _, d, g in sorted(cands, reverse=True):
            d, g = _echo_refine(LM[0], WW[0], d, g); g = float(np.clip(g, -0.95, 0.95)); cl = comb_db(d, g) * (np.log(10) / 20); k = int(round(d))
            if np.abs(_ceps(LM[0] - cl, WW[0])[max(_QLO, k - 10):k + 11]).max() <= 0.5 * abs(g): got = (d, g, cl); break     # 4
        if got is None: break
        out.append(got[:2]); LM = [L - got[2] for L in LM]
    return out

# ------------------------------------------------------------------------------------------------ the blind curve
def _level_offset(curve, pm):
    """dB to add to the curve so that taking it out (with the default caps) leaves the sound's long-term power as it is."""
    p = 10 ** ((pm - pm.max()) / 10); g = _soft_cap(-curve); return 10 * np.log10(max((p * 10 ** (g / 10)).sum() / p.sum(), 1e-12))

def _analyse_view(Q, echo=True, model=True, sens=1.0):
    """One view of the sound (a channel, or both pooled) -> the raw parts: reflections, the model's curve on GRID (its weighted median
    taken out, not yet shrunk) and its doubt per cell, which cells carry sound, and the long-term power spectrum (dB per bin)."""
    taps = find_echoes(Q, sens=sens) if echo else []; mc = np.zeros(G); v = np.ones(G)
    o = _bin_stats(Q) if model else dict(pmean=10 * np.log10(np.maximum(Q.mean(0), 1e-30)))             # the statistics are only needed for the model
    if model:
        if taps:                                                    # the model looks at the sound without the reflections already found
            cb = sum(comb_db(d, g) for d, g in taps); o = dict(o); o['lmean'] = o['lmean'] - cb; o['floor'] = o['floor'] - ndi.uniform_filter1d(cb, _FW, mode='nearest')
        F = _grid_feats(o); v = _valid(F); mc, doubt = _net(_net_input(F)); mc = mc - _wmedian(mc, v)
    else: doubt = np.ones(G)
    return dict(echoes=taps, model_raw=mc, doubt=doubt, valid=v, pmean=o['pmean'])

def _finish_view(part, restraint, min_db=MODEL_MIN_DB, detail=1.0):
    """Raw parts -> the curve on the bins: reflections' comb curves plus the shrunk model curve (smoothed by Detail; the comb curves are
    not: smoothing would only erase their teeth). Outside the sound's own frequency range the model curve holds its last value; a model
    curve that stays under min_db everywhere counts as nothing. The level is set so that taking the curve out leaves the long-term
    power as it is. Nothing found -> exactly zero."""
    curve = np.zeros(len(FB))
    for d, g in part['echoes']: curve += comb_db(d, g)
    mc = _shrink(part['model_raw'], restraint * part['doubt']); i = np.flatnonzero(part['valid'] > 0.5)
    if len(i): mc = mc.copy(); mc[:i[0]] = mc[i[0]]; mc[i[-1] + 1:] = mc[i[-1]]
    if not len(i) or np.abs(mc).max() < min_db: mc = np.zeros(G)
    part['model_db'] = mc
    if not part['echoes'] and not mc.any(): return curve
    if mc.any(): curve = curve + smooth(FB, _to_bins(GRID, mc), detail)
    return curve + _level_offset(curve, part['pmean'])

def contour(x, detail=1.0, stereo='linked', echo=True, model=False, restraint=None, sens=1.0, info=None):
    """The tone curve found on the sound x, blind. Returns (grid_hz, curve_db): curve_db has one row ('linked': both channels pooled into
    one curve) or two ('dual': left and right on their own). All zeros = nothing found.
    echo: the echo finder (reflections); sens: its sensitivity (1 = default; 1.5 accepts reflections two thirds as strong).
    model: add the model's guess for everything that is not a reflection (EQ, filter, cabinet ...); off by default, see the top of the file.
    restraint: how hard the model's curve is shrunk, in units of its own doubt per cell (None = the value stored with the model, 2; 0 = not at all).
    detail: see smooth(); it acts on the model's part only. A reflection is kept exactly as found, teeth and all.
    info: pass a list to get the parts per view: 'echoes' [(delay in samples, gain)], 'model_db' (the model part on GRID as used),
    'model_raw' and 'doubt' (before shrinking; the doubt is the error in dB the model itself expects per cell)."""
    P = _frames(x); a = _active(P); P = P[:, a, :]
    if P.shape[1] < 2: return FB.copy(), (np.zeros(len(FB)) if stereo == 'linked' else np.zeros((2, len(FB))))          # silence
    if restraint is None: restraint = float(_load_model()['restraint']) if model else 0.0
    views = [P[0] + P[1]] if stereo == 'linked' else [P[0], P[1]]; parts = [_analyse_view(Q, echo, model, sens) for Q in views]
    c = np.stack([_finish_view(pt, restraint, MODEL_MIN_DB, detail) for pt in parts])
    if info is not None: info.extend(parts)
    return FB.copy(), (c[0] if stereo == 'linked' else c)

# ------------------------------------------------------------------------------------------------ Detail
def detail_sigma(detail):
    """Detail 0..1 -> width (Gaussian sigma, octaves) of the smoothing: 0 -> 1 octave, 0.5 -> 1/8, 0.75 -> 1/23, 1 -> none."""
    return 0.0 if detail >= 1.0 else 2.0 ** (-6.0 * max(detail, 0.0))

def _smooth_lin(v, sigma_oct):
    """Smoothing by a Gaussian that is sigma_oct octaves wide everywhere, for a curve on evenly spaced bins (bin k at k * step):
    three passes of a running mean whose half width grows with the bin number."""
    n = len(v); k = np.arange(n, dtype=float); sig = sigma_oct * np.log(2.0) * k; h = 0.5 * (np.sqrt(4 * sig * sig + 1) - 1); pos = np.arange(n + 1) - 0.5
    for _ in range(3):
        cs = np.concatenate([[0.0], np.cumsum(v)]); a = np.clip(k - h - 0.5, -0.5, n - 0.5); b = np.clip(k + h + 0.5, -0.5, n - 0.5)
        v = (np.interp(b, pos, cs) - np.interp(a, pos, cs)) / np.maximum(b - a, 1e-9)
    return v

def smooth(grid_hz, curve_db, detail):
    """The Detail control: the curve smoothed along log frequency (see detail_sigma). detail 1 returns the curve as it is. Works on the
    curve's own grid when that is evenly spaced (the engine's bins); other grids are smoothed on a 1/96-octave axis."""
    curve_db = np.asarray(curve_db, float); s = detail_sigma(detail)
    if s == 0.0: return curve_db.copy()
    if curve_db.ndim == 2: return np.stack([smooth(grid_hz, c, detail) for c in curve_db])
    g = np.asarray(grid_hz, float); d = np.diff(g)
    if np.allclose(d, d[0], rtol=1e-6) and abs(g[0]) < 1e-9: return _smooth_lin(curve_db, s)
    ax = 10.0 * 2.0 ** (np.arange(int(np.log2(2400.0) * 96) + 1) / 96.0); v = _to_bins(g, curve_db, ax); sg = s * 96.0; kk = int(np.ceil(4 * sg)); t = np.arange(-kk, kk + 1); w = np.exp(-0.5 * (t / sg) ** 2); w /= w.sum()
    v = np.convolve(np.concatenate([np.full(kk, v[0]), v, np.full(kk, v[-1])]), w, 'valid'); return np.interp(np.log(np.maximum(g, 1e-3)), np.log(ax), v)

# ------------------------------------------------------------------------------------------------ curve -> filter
def _soft_cap(g, boost=MAX_BOOST_DB, cut=MAX_CUT_DB, knee=0.75):
    """Gains beyond the caps are bent in softly: untouched up to knee * cap, then levelling out at the cap."""
    def one(v, cap): k = knee * cap; return np.where(v <= k, v, k + (cap - k) * np.tanh((v - k) / max(cap - k, 1e-9)))
    return np.where(g >= 0, one(np.maximum(g, 0), boost), -one(np.maximum(-g, 0), cut))

def _minphase(mag_db, nfft):
    """Minimum-phase impulse response with the given magnitude (dB on nfft/2+1 bins): fold the cepstrum of the log magnitude."""
    c = np.fft.irfft(mag_db * (np.log(10.0) / 20.0), nfft); w = np.zeros(nfft); w[0] = 1.0; w[1:nfft // 2] = 2.0; w[nfft // 2] = 1.0
    return np.fft.irfft(np.exp(np.fft.rfft(c * w)), nfft)

def to_ir(grid_hz, curve_db, n=16384, phase='minimum', sr=SR):
    """The curve as an impulse response of n taps: (n,) for one curve, (n, 2) for two rows (left, right).
    'minimum': minimum phase, starts at tap 0 (no latency; the exact inverse of a minimum-phase effect such as an EQ or a single reflection).
    'linear': symmetric around tap n // 2 (latency n // 2 samples). No caps are applied here: the curve is taken as given."""
    curve_db = np.asarray(curve_db, float)
    if curve_db.ndim == 2: return np.stack([to_ir(grid_hz, c, n, phase, sr) for c in curve_db], 1)
    nfft = int(2 ** np.ceil(np.log2(4 * n))); db = _to_bins(grid_hz, curve_db, np.fft.rfftfreq(nfft, 1.0 / sr))
    if phase == 'minimum':
        h = _minphase(db, nfft)[:n].copy(); k = max(8, n // 8); h[-k:] *= 0.5 * (1 + np.cos(np.pi * (np.arange(k) + 0.5) / k)); return h
    h = np.roll(np.fft.irfft(10.0 ** (db / 20.0), nfft), n // 2)[:n].copy(); k = n // 4; r = 0.5 * (1 - np.cos(np.pi * (np.arange(k) + 0.5) / k)); h[:k] *= r; h[-k:] *= r[::-1]; return h

def filter_ir(x, h, phase='minimum'):
    """x (n, 2) through the impulse response h ((taps,) or (taps, 2)); same length out, the linear-phase latency taken out."""
    x = _stereo(x)
    if h.ndim == 1: h = np.stack([h, h], 1)
    d = 0 if phase == 'minimum' else h.shape[0] // 2
    return np.stack([ss.fftconvolve(x[:, c], h[:, c])[d:d + len(x)] for c in range(2)], 1)

def _gain_filter(x, grid_hz, gain_db, phase, n):
    gain_db = np.asarray(gain_db, float)
    if not gain_db.any(): return _stereo(x).copy()                            # nothing to do: the sound itself, exactly
    return filter_ir(x, to_ir(grid_hz, _soft_cap(gain_db), n, phase), phase)

def apply_curve(x, grid_hz, curve_db, amount=0.0, phase='minimum', n=16384):
    """The fader on the analysed sound: gain = (amount - 1) * curve. amount 0 = the curve taken out (flattened), 1 = x as it is,
    2 = the curve applied once more. Gains are capped softly at +MAX_BOOST_DB / -MAX_CUT_DB."""
    if amount == 1.0: return _stereo(x).copy()
    return _gain_filter(x, grid_hz, (amount - 1.0) * np.asarray(curve_db, float), phase, n)

def put_curve(y, grid_hz, curve_db, amount=1.0, phase='minimum', n=16384):
    """A curve (found on another sound, or from match) onto the sound y: gain = amount * curve. 0 = none, 1 = as found, 2 = double."""
    if amount == 0.0: return _stereo(y).copy()
    return _gain_filter(y, grid_hz, amount * np.asarray(curve_db, float), phase, n)

# ------------------------------------------------------------------------------------------------ Tone Match
def long_term_db(x, stat='pmean', stereo='linked'):
    """Long-term spectrum of a sound on the bins, dB, silent frames left out: one row (both channels pooled) or two ('dual').
    'pmean': power mean over the frames (the usual long-term average spectrum); 'lmean': mean of the clamped log power."""
    P = _frames(x); a = _active(P); views = [(P[0] + P[1])[a]] if stereo == 'linked' else [P[0][a], P[1][a]]
    if not a.any(): return np.full(len(FB), -300.0) if stereo == 'linked' else np.full((2, len(FB)), -300.0)             # silence
    S = np.stack([10 * np.log10(np.maximum(Q.mean(0), 1e-30)) if stat == 'pmean' else _clamped_log(Q)[0].mean(0) for Q in views])
    return S[0] if stereo == 'linked' else S

def _match_one(Sa, Sb, detail, max_db):
    k = np.ones(9) / 9; va = np.convolve(Sa, k, 'same'); vb = np.convolve(Sb, k, 'same'); ok = (va > va.max() - 60) & (vb > vb.max() - 60) & (FB >= 20)
    i = np.flatnonzero(ok)
    if not len(i): return np.zeros(len(FB))
    c = np.interp(np.arange(len(FB)), i, (Sa - Sb)[i]); c = smooth(FB, c, detail); c = np.clip(c - np.median(c[i]), -max_db, max_db); p = 10 ** ((Sb - Sb.max()) / 10)
    for _ in range(3): c = c - 10 * np.log10(max((p * 10 ** (_soft_cap(c) / 10)).sum() / p.sum(), 1e-12))        # b keeps its long-term power (at amount 1, caps included)
    return c

def match(a, b, detail=0.6, max_db=30.0, stat='pmean', stereo='linked'):
    """Tone Match: the curve (grid_hz, curve_db) that gives sound b the long-term spectrum of sound a; apply with put_curve(b, ...).
    The difference of the two long-term spectra, smoothed by Detail (0.6 for two different sounds; 1 when a and b are two versions of
    the same recording). It is only taken where both sounds have something (within 60 dB of their loudest region); outside, the
    nearest valid value is held. Values beyond +-max_db are cut, and the level is set so that b keeps its long-term power when the
    curve is put on at amount 1. stereo='dual' gives one curve per channel (two rows)."""
    Sa = long_term_db(a, stat, stereo); Sb = long_term_db(b, stat, stereo)
    if Sa.max() < -250 or Sb.max() < -250: return FB.copy(), np.zeros_like(Sa)                                           # one of them is silent
    if stereo == 'linked': return FB.copy(), _match_one(Sa, Sb, detail, max_db)
    return FB.copy(), np.stack([_match_one(Sa[c], Sb[c], detail, max_db) for c in range(2)])
