# Level movement engine, prototype: tremolo, auto pan and gated rhythm are a gain per channel that moves over time and repeats.
# It is measured on ONE sound (period, shape over one cycle, depth, how left and right differ) and used both ways:
#   find(x)            what was measured, or None
#   apply(x, F, s)     s = -1 takes it out of the sound it was found on, s = +1 puts it on another sound; in between and up to +2 scales it in dB
# How a gain put on the sound is told from the sound's own rhythm: a gain moves ALL bands by the same amount in dB at the same
# moment and repeats exactly; notes and hits do not.
import numpy as np, scipy.fft as sfft
SR = 48000; W = 256; HOP = 48                                              # 5.3 ms windows, one every millisecond
EDG = [0, 2, 3, 4, 5, 7, 9, 12, 16, 21, 28, 37, 49, 64, 85, 108]          # band edges in bins of 187.5 Hz (15 bands up to 20 kHz)
NB = len(EDG) - 1
FLOOR = 1e-10                                                              # a band counts as silent 100 dB under its loudest moment

# ---------------------------------------------------------------- what is looked at
def band_power(x1):
    """Power per band in frames of W samples centred on every HOP-th sample: (frames, NB)."""
    n = len(x1); T = n // HOP + 1; pad = np.concatenate([np.zeros(W // 2), x1, np.zeros(W)]); win = np.hanning(W + 1)[:W]; out = np.empty((T, NB))
    for a in range(0, T, 4096):
        idx = np.arange(a, min(T, a + 4096))[:, None] * HOP + np.arange(W)[None, :]; Z = sfft.rfft(pad[idx] * win, axis=1); P = Z.real ** 2 + Z.imag ** 2
        cs = np.concatenate([np.zeros((len(P), 1)), np.cumsum(P, 1)], 1); out[a:a + len(P)] = cs[:, EDG[1:]] - cs[:, EDG[:-1]]
    return out
def levels(x):
    """Both channels: (2, frames, NB)."""
    return np.stack([band_power(x[:, 0]), band_power(x[:, 1])])
def box(v, m):
    """Mean over m frames centred on each frame (zeros outside), along the first axis."""
    if m <= 1: return v
    cs = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, 0)]); T = len(v); a = np.clip(np.arange(T) - m // 2, 0, T); b = np.clip(np.arange(T) - m // 2 + m, 0, T)
    return (cs[b] - cs[a]) / m
def lanes(E, m=16):
    """Two lanes over time, in dB, one value per frame:
       u = the level both channels and all bands share (median over bands of each band's level against its own mean over the sound),
       d = left minus right (median over the bands that sound on both sides).  NaN where nothing sounds."""
    S = box(E[0], m) + 0.0, box(E[1], m) + 0.0; tot = S[0] + S[1]; ref = tot.mean(0); live = ref > ref.max() * 1e-7; top = tot.max(0)
    on = (tot > top * 1e-6) & live[None, :]                                # the band sounds here (60 dB under its loudest moment or more)
    L = 10 * np.log10(np.maximum(tot, top * 1e-9 + 1e-300) / np.maximum(ref, 1e-300)); L[~on] = np.nan
    D = 10 * np.log10(np.maximum(S[0], 1e-300) / np.maximum(S[1], 1e-300)); D = np.clip(D, -60, 60); D[~on] = np.nan
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); u = np.nanmedian(L, 1); d = np.nanmedian(D, 1)
    cnt = on.sum(1); u[cnt < 3] = np.nan; d[cnt < 3] = np.nan
    return u, d

# ---------------------------------------------------------------- the period
def periodogram_peaks(v, fs, flo, fhi, most=6):
    """Strongest lines of a lane between flo and fhi Hz: [(Hz, share of the lane's power)], strongest first. Holes (NaN) count as the lane's median."""
    ok = np.isfinite(v)
    if ok.sum() < 64: return []
    med = np.median(v[ok]); z = np.where(ok, v - med, 0.0); lo, hi = np.percentile(z[ok], [2, 98]); z = np.clip(z, lo, hi); z = z - z.mean()
    n = len(z); nfft = 1 << int(np.ceil(np.log2(8 * n))); F = np.abs(sfft.rfft(z * np.hanning(n), nfft)) ** 2; f = np.arange(len(F)) * fs / nfft
    tot = (z ** 2 * np.hanning(n) ** 2).sum() * nfft / 2 + 1e-30
    a, b = int(np.ceil(flo / fs * nfft)), int(min(len(F) - 2, np.floor(fhi / fs * nfft)))
    if b <= a + 2: return []
    P = F.copy(); P[:a] = 0; P[b + 1:] = 0; out = []; wid = int(np.ceil(1.6 * nfft / n))          # half width of a line (Hann window)
    for _ in range(most):
        k = int(np.argmax(P))
        if P[k] <= 0: break
        y0, y1, y2 = F[k - 1], F[k], F[k + 1]; den = y0 - 2 * y1 + y2; off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        out.append(((k + off) * fs / nfft, float(F[max(0, k - wid):k + wid + 1].sum() / tot))); P[max(0, k - 2 * wid):k + 2 * wid + 1] = 0
    return out
def diff_lane(E, tau, m=4):
    """What all bands do together between t and t + tau frames: the median over bands (both channels) of the level change in dB,
    one value per frame (NaN where fewer than three bands sound at both moments)."""
    import warnings
    S = np.concatenate([box(E[0], m), box(E[1], m)], 1); top = S.max(0); live = top > top.max() * 1e-7; S = S[:, live]; top = top[live]
    on = S > top * FLOOR; L = 10 * np.log10(np.maximum(S, top * FLOOR * 0.01 + 1e-300))
    d = L[tau:] - L[:-tau]; both = on[tau:] | on[:-tau]; d[~both] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); v = np.nanmedian(d, 1)
    v[both.sum(1) < 3] = np.nan; return v

def cell_levels(E, Pf, K, t0=0.0):
    """Band levels gathered into cells of a K-th of the period Pf (frames): dB of the mean power per cell, for every band of both
    channels: (cells, 2 NB), cells = whole cycles * K. Below the band's floor: NaN."""
    T = E.shape[1]; pos = (np.arange(T) - t0) / Pf * K; idx = np.floor(pos).astype(np.int64); n = (idx.max() + 1) // K * K; ok = (idx >= 0) & (idx < n)
    if n < 2 * K: return np.zeros((0, 2 * NB))
    cnt = np.maximum(np.bincount(idx[ok], minlength=n), 1).astype(float); C = np.empty((n, 2 * NB))
    for ch in range(2):
        for b in range(NB): C[:, ch * NB + b] = np.bincount(idx[ok], weights=E[ch][ok, b], minlength=n) / cnt
    top = C.max(0); L = 10 * np.log10(np.maximum(C, 1e-300)); L[C <= top * FLOOR] = np.nan; L[:, top <= top.max() * 1e-7] = np.nan
    return L

def pairs_fold(L, K, lags=None, use=None):
    """The shape over one cycle that all bands share, from level differences: for several lags j, the median over bands and cycles of
    level(k + j) - level(k) says how the gain differs between two phases (a note that starts or ends upsets only the few pairs that
    straddle it); the shape is the least squares solution of all those medians. Returns (shape in dB with mean 0 (K,), share of the
    differences' spread it explains, number of pairs)."""
    lags = lags or [j for j in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if j <= K // 2]; n = len(L); fl = np.isnan(L)
    if n < 2 * K: return np.zeros(K), 0.0, 0
    low = np.nanmin(np.where(fl, np.inf, L), 0); low[~np.isfinite(low)] = 0.0; Lf = np.where(fl, low[None, :] - 20.0, L)      # silent: lower than anything heard
    A = np.zeros((K, K)); rhs = np.zeros(K); keep = []; ar = np.arange(K)
    for j in lags:
        d = Lf[j:] - Lf[:-j]; d[fl[j:] & fl[:-j]] = np.nan; m = (n - j) // K * K                     # whole cycles of pairs
        if use is not None: d[~(use[j:] & use[:-j])] = np.nan
        if m < K: continue
        D = d[:m].reshape(m // K, K, -1).transpose(1, 0, 2).reshape(K, -1)                           # (phase, cycles x bands)
        cnt = np.isfinite(D).sum(1); med = np.zeros(K); mad = np.ones(K)
        good = cnt >= 3
        if not good.any(): continue
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); med[good] = np.nanmedian(D[good], 1); mad[good] = np.nanmedian(np.abs(D[good] - med[good, None]), 1) * 1.4826
        if good.all(): med = med - med.mean()                # a shape that repeats has differences that add up to nothing round the cycle: what is left over is the sound's own drift (notes dying away)
        w = np.where(good, cnt / np.maximum(mad, 0.3) ** 2, 0.0); b = (ar + j) % K
        np.add.at(A, (ar, ar), w); np.add.at(A, (b, b), w); np.add.at(A, (ar, b), -w); np.add.at(A, (b, ar), -w); np.add.at(rhs, b, w * med); np.add.at(rhs, ar, -w * med)
        keep.append((j, D, b))
    if not keep: return np.zeros(K), 0.0, 0
    A += 1e-9 * np.trace(A) / K * np.ones((K, K)); g = np.linalg.lstsq(A, rhs, rcond=None)[0]; g -= g.mean()
    num = den = 0.0; npairs = 0
    for j, D, b in keep:                                    # how much of the differences' spread does the shape explain (absolute values: robust)
        ok = np.isfinite(D); pred = (g[b] - g)[:, None]; c = np.nanmedian(D)
        num += np.abs(np.where(ok, np.clip(D - pred, -40, 40), 0)).sum(); den += np.abs(np.where(ok, np.clip(D - c, -40, 40), 0)).sum(); npairs += int(ok.sum())
    return g, float(1.0 - num / max(den, 1e-9)), npairs

def fold_stat(v, P, K=16):
    """Epoch folding: the share of lane v's power that repeats with period P (frames), seen in K phase bins. v: clipped, zero where unknown; w: 1 where known."""
    n = len(v); idx = np.floor((np.arange(n) % P) / P * K).astype(np.int64); idx[idx >= K] = K - 1
    s = np.bincount(idx, weights=v, minlength=K); c = np.bincount(idx, minlength=K); return float((s * s / np.maximum(c, 1)).sum())

def clip_lane(v, k=1.5):
    """A lane made safe for sums: its median taken off, everything beyond k times its usual size (median absolute value) pulled in,
    unknown (NaN) set to 0. What is left says little more than 'up' or 'down': a steady small movement then counts as much as a loud hit.
    Returns (lane, its power, known mask)."""
    ok = np.isfinite(v)
    if ok.sum() < 32: return np.zeros(len(v)), 0.0, ok
    med = np.median(v[ok]); c = np.median(np.abs(v[ok] - med)) * k + 1e-9; z = np.clip(np.where(ok, v - med, 0.0), -c, c); z[ok] -= z[ok].mean(); z[~ok] = 0.0
    return z, float((z ** 2).sum()), ok

TAUS = (4, 8, 16, 32, 64, 128, 256, 512)
def scan_together(E, pmin=40.0, pmax=None, step=0.002, K=16):
    """For every period on a fine grid (frames): how much of what all bands do together repeats with it. The lane used for a
    period P is the level change over about a third of P. Returns (periods, share explained, excess over what any period explains)."""
    T = E.shape[1]; pmax = pmax or T / 3.0; Ps = np.exp(np.arange(np.log(pmin), np.log(pmax), step)); S = np.zeros(len(Ps)); lane = {}
    for i, P in enumerate(Ps):
        tau = max(t for t in TAUS if t <= max(P / 2.5, TAUS[0]))
        if tau not in lane: lane[tau] = clip_lane(diff_lane(E, tau, max(2, tau // 4)))
        z, pw, ok = lane[tau]
        if pw > 0: S[i] = fold_stat(z, P, K) / pw
    return Ps, S, lane

def peaks_of(Ps, S, most=6, span=0.2):
    """Periods where the scan stands out of its surroundings (within +-span in log period): [(period, share, excess)], largest excess first."""
    import scipy.ndimage as ndi
    step = np.log(Ps[1] / Ps[0]); w = int(span / step) | 1; base = ndi.median_filter(S, 2 * w + 1, mode='nearest'); ex = S - base; out = []; e = ex.copy()
    for _ in range(most):
        k = int(np.argmax(e))
        if e[k] <= 0.01: break
        out.append((float(Ps[k]), float(S[k]), float(ex[k]))); a = int(0.03 / step); e[max(0, k - a):k + a + 1] = 0
    return out

def refine_period(z, P, K=32, span=0.004, steps=41):
    """The period near P (frames) that makes lane z repeat best (a fine scan, then the top of a parabola through the three best points)."""
    Ps = P * (1 + np.linspace(-span, span, steps)); s = np.array([fold_stat(z, p, K) for p in Ps]); k = int(np.argmax(s))
    if 0 < k < steps - 1:
        den = s[k - 1] - 2 * s[k] + s[k + 1]; off = 0.5 * (s[k - 1] - s[k + 1]) / den if den < 0 else 0.0; return float(Ps[k] + off * (Ps[1] - Ps[0]))
    return float(Ps[k])

CLAMP = 30.0                                                               # for the agreement measures a level more than this under the top counts as this far under it
def agreement(L, K, g, start=0, skip=()):
    """Do all bands, and all cycles, carry the shared shape g? Levels are taken against each band's top in the cycle and clamped CLAMP dB
    under it (so a gate that closes fully can be compared too). Returns dict:
      slopes, corrs   per band (of both channels) its own shape over the cycle (median over cycles) set against g: size and likeness
      sizes           per cycle what the bands share (median over bands) set against g: size (1 = as deep as g)
      halves          likeness of the shared shape of the even and of the odd cycles
      r2c             share of the shared level's movement that g explains
      silent          share of cells in which nothing sounds"""
    import warnings
    n = (len(L) - start) // K; q = lambda v, p: float(np.percentile(v, p)) if len(v) else 0.0
    if n < 2: return None
    sk = set(skip); keep = np.array([i for i in range(n) if i not in sk], dtype=np.int64)
    if len(keep) < 2: return None
    C = L[start:start + n * K].reshape(n, K, -1)[keep]; fl = np.isnan(C); snd = (~fl).any(1, keepdims=True)                   # (cycles, 1, bands): the band sounds somewhere in the cycle
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        top = np.nanpercentile(C, 90, axis=1, keepdims=True); dev = np.where(fl, -CLAMP, np.maximum(C - top, -CLAMP)); dev = np.where(np.broadcast_to(snd, dev.shape), dev, np.nan)
        own = np.nanmedian(dev, 0); com = np.nanmedian(dev, 2)                                                                 # (K, bands) and (cycles, K)
        gl = np.roll(g, -start); gc = np.maximum(gl - np.sort(gl)[-max(1, K // 8):].mean(), -CLAMP); gg = gc - gc.mean(); den = (gg ** 2).sum() + 1e-12
        sl = []; co = []; ub = []
        for b in range(own.shape[1]):
            v = own[:, b]
            if not np.isfinite(v).all() or snd[:, 0, b].mean() < 0.3: continue
            v = v - v.mean(); sl.append(float((v * gg).sum() / den)); co.append(float((v * gg).sum() / np.sqrt(den * ((v ** 2).sum() + 1e-12)))); ub.append(b)
        dis_med = dis_hi = 1.0                                                  # how far the bands' own shapes lie apart, phase by phase, against the depth
        if len(ub) >= 3:
            o = own[:, ub]; dis = np.median(np.abs(o - np.median(o, 1, keepdims=True)), 1); depc = max(float(-gc.min()), 1e-6); dis_med = float(np.median(dis) / depc); dis_hi = float(np.percentile(dis, 90) / depc)
        sz = []
        for i in range(len(keep)):
            v = com[i]
            if not np.isfinite(v).all(): continue
            v = v - v.mean(); sz.append(float((v * gg).sum() / den))
        ev, od = np.nanmedian(com[0::2], 0), np.nanmedian(com[1::2], 0); okh = np.isfinite(ev) & np.isfinite(od)
        halves = float(np.corrcoef(ev[okh], od[okh])[0, 1]) if okh.sum() > K // 2 and np.std(ev[okh]) > 1e-9 and np.std(od[okh]) > 1e-9 else 0.0
        okc = np.isfinite(com); cm = com - np.nanmean(com, 1, keepdims=True); num = np.abs(np.where(okc, cm - gg[None, :], 0)).sum(); den2 = np.abs(np.where(okc, cm, 0)).sum()
    return dict(bands=len(sl), slope_med=q(sl, 50), slope_lo=q(sl, 25), slope_hi=q(sl, 75), slope_min=q(sl, 10), slope_max=q(sl, 90), corr_med=q(co, 50), corr_lo=q(co, 25), corr_min=q(co, 10),
                cyc_med=q(sz, 50), cyc_lo=q(sz, 25), cyc_hi=q(sz, 75), cyc_n=len(sz), halves=halves, r2c=float(1 - num / max(den2, 1e-9)), silent=float(fl.all(2).mean()), dis_med=dis_med, dis_hi=dis_hi)

def shape_numbers(g):
    """A cycle shape in a few numbers: up = share of the cycle spent above the middle between top and bottom, steep = the largest step
    between neighbouring bins as a share of the depth, fund = share of the shape's variance in its first harmonic, rise = share of the
    cycle it takes from the bottom to the top (0.5: even, small: jumps up and sinks back like a struck note)."""
    K = len(g); lo, hi = g.min(), g.max(); dep = max(hi - lo, 1e-9); up = float((g > 0.5 * (lo + hi)).mean()); steep = float(np.abs(np.diff(np.concatenate([g, g[:1]]))).max() / dep)
    F = np.abs(np.fft.rfft(g - g.mean())) ** 2; fund = float(F[1] / max(F[1:].sum(), 1e-12)); rise = float(((int(np.argmax(g)) - int(np.argmin(g))) % K) / K)
    return up, steep, fund, rise

def struck_numbers(g):
    """How a shape gets from one level to the other: (share of the cycle spent falling through the middle, between a fifth of the depth
    under the top and a fifth over the bottom; the same rising; share of the cycle within 3 dB of the top). A struck note jumps up and
    sinks back (falling long, rising short); what an effect does to a sound is as a rule as quick one way as the other."""
    K = len(g); srt = np.sort(g); k = max(1, K // 16); lo = srt[:k].mean(); hi = srt[-k:].mean(); dep = max(hi - lo, 1e-9)
    mid = (g > lo + 0.2 * dep) & (g < hi - 0.2 * dep); sl = np.roll(g, -1) - np.roll(g, 1)
    return float((mid & (sl < 0)).mean()), float((mid & (sl > 0)).mean()), float((g >= hi - 3.0).mean())

def top_of(g): return float(np.sort(g)[-max(1, len(g) // 8):].mean())
def depth_of(g): srt = np.sort(g); k = max(1, len(g) // 16); return float(srt[-k:].mean() - srt[:k].mean())

def measure_together(E, P, lane=None):
    """Everything measured about 'all bands of both channels move together with period P (frames)': the shape and how sure it is."""
    if lane is not None and lane[1] > 0: P = refine_period(lane[0], P)
    K = int(np.clip(round(P / 2), 12, 48)); L = cell_levels(E, P, K); g, r2, npairs = pairs_fold(L, K)
    if npairs == 0: return None
    start, skip = 0, []; up, steep, fund, rise = shape_numbers(g); n = len(L) // K
    if steep >= 0.3 and depth_of(g) >= 6.0:                                    # a gate's shape: are steps left out? then the shape from the others
        g, start, skip = gate_steps(L, K, g); up, steep, fund, rise = shape_numbers(g)
    a = agreement(L, K, g, start, skip)
    if a is None: return None
    fall, climb, flat = struck_numbers(g)
    m = dict(P=float(P), K=K, g=g, start=int(start), skip=[int(i) for i in skip], r2=r2, depth=depth_of(g), cycles=n, skipfrac=len(skip) / max(n, 1), up=up, steep=steep, fund=fund, rise=rise, fall=fall, climb=climb, flat=flat); m.update(a); return m

PMIN = 60.0                                                                # movements faster than 16.7 Hz are not looked for: up there a sound's own roughness and beating take over
def candidates_together(E, most=12):
    Ps, S, lane = scan_together(E, pmin=PMIN); out = []
    for P, share, excess in peaks_of(Ps, S, most):
        tau = max(t for t in TAUS if t <= max(P / 2.5, TAUS[0])); m = measure_together(E, P, lane.get(tau))
        if m is None: continue
        m['share'] = share; m['excess'] = excess; out.append(m)
    return out

# ---------------------------------------------------------------- a gate that leaves steps out
def plateaus(g):
    """A two-level shape: (first bin of the cycle laid out so that it starts in the middle of the low stretch, bins of the high stretch,
    bins of the low stretch before it, bins of the low stretch after it), all as offsets from that first bin; None if g is not two-level."""
    K = len(g); lo, hi = g.min(), g.max(); dep = hi - lo
    if dep < 6.0: return None
    high = g > hi - 0.25 * dep; low = g < lo + 0.25 * dep
    if high.sum() < 2 or low.sum() < 2 or (high | low).mean() < 0.7: return None
    d = np.diff(np.concatenate([g, g[:1]])); kr = int(np.argmax(d)); kf = int(np.argmin(d))                      # the step up is between kr and kr + 1, down between kf and kf + 1
    nlow = (kr - kf) % K; start = (kf + 1 + nlow // 2) % K; rel = (np.arange(K) - start) % K                      # rel: place of each bin in the laid-out cycle
    up_at = (kr + 1 - start) % K; down_at = (kf + 1 - start) % K
    if not 0 < up_at < down_at < K: return None
    order = np.argsort(rel); hi_b = [int(r) for r in range(up_at, down_at) if high[order[r]]]; lo_a = [int(r) for r in range(0, up_at) if low[order[r]]]; lo_b = [int(r) for r in range(down_at, K) if low[order[r]]]
    if len(hi_b) < 1 or len(lo_a) + len(lo_b) < 1: return None
    return start, hi_b, lo_a, lo_b

def left_out(L, K, g, share=0.4):
    """Steps a gate leaves out: cycles in which the level neither steps up nor steps down where the shape g says it should (looked at
    right at the two edges, two bins either side, median over the bands that sound there). Returns (start bin of the laid-out cycles,
    list of left-out cycles counted from there) or None when g is not a gate's shape."""
    import warnings
    pl = plateaus(g)
    if pl is None: return None
    start, hi_b, lo_a, lo_b = pl; n = (len(L) - start) // K
    if n < 3: return None
    fl = np.isnan(L); low = np.nanmin(np.where(fl, np.inf, L), 0); low[~np.isfinite(low)] = 0.0; Lf = np.where(fl, low[None, :] - 20.0, L)
    C = Lf[start:start + n * K].reshape(n, K, -1); S = ~fl[start:start + n * K].reshape(n, K, -1); dep = g.max() - g.min(); out = []
    def edge(hi, lo):
        if not hi or not lo: return np.full(n, np.nan)
        d = C[:, hi, :].mean(1) - C[:, lo, :].mean(1); snd = S[:, hi, :].any(1) | S[:, lo, :].any(1); d[~snd] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); r = np.nanmedian(d, 1)
        r[snd.sum(1) < 3] = np.nan; return r
    up = edge(hi_b[:2], lo_a[-2:]); down = edge(hi_b[-2:], lo_b[:2])
    for i in range(n):
        v = [x for x in (up[i], down[i]) if np.isfinite(x)]
        if v and max(v) < share * dep: out.append(i)
    return start, out

def gate_steps(L, K, g):
    """A gate's shape measured again from the steps it does not leave out. Returns (shape, start bin of the laid-out cycles, left-out cycles)."""
    skip = []; start = 0
    for rnd in range(2):
        lo = left_out(L, K, g, 0.3 if rnd == 0 else 0.4)
        if lo is None: return g, 0, []
        start, skip = lo
        if not skip or len(skip) > 0.7 * ((len(L) - start) // K): return g, start, []
        use = np.ones(len(L), bool)
        for i in skip: use[start + i * K:start + (i + 1) * K] = False
        g2, r2, npairs = pairs_fold(L, K, use=use)
        if npairs == 0: return g, start, []
        g = g2
    return g, start, skip

# ---------------------------------------------------------------- left against right (auto pan)
def law_gains(d, p):
    """What each side does (dB) when left is d dB above right and the two follow the law gL^p + gR^p = 2 (p = 1: the gains add up to a
    constant, p = 2: the powers do, p -> 0: as many dB up on one side as down on the other)."""
    r = 10 ** (np.clip(d, -80, 80) / 20.0)
    if p < 0.05: return d / 2.0, -d / 2.0
    c = (20.0 / p) * (np.log10(2.0) - np.log10(1.0 + r ** p)); return c + d, c

def pan_lane(E, m=16):
    """Left minus right in dB over time (median over the bands that sound on both sides), and per band."""
    import warnings
    S0, S1 = box(E[0], m), box(E[1], m); tot = S0 + S1; top = tot.max(0); live = top > top.max() * 1e-7; on = (tot > top * 1e-5) & live[None, :]
    D = np.clip(10 * np.log10(np.maximum(S0, 1e-300) / np.maximum(S1, 1e-300)), -60, 60); D[~on] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); own = np.nanmedian(D, 0); own[~np.isfinite(own)] = 0.0; d = np.nanmedian(D - own[None, :], 1)      # each band against its own usual place in the picture
        d = d + np.nanmedian(own[live]) if live.any() else d
    d[on.sum(1) < 3] = np.nan; return d, D

def fold_profile(v, P, K, stat='median'):
    """One cycle of lane v (NaN = unknown) at period P frames in K bins; bins nothing fell into are filled in from their neighbours."""
    n = len(v); idx = np.floor((np.arange(n) % P) / P * K).astype(np.int64); idx[idx >= K] = K - 1; prof = np.full(K, np.nan)
    for k in range(K):
        w = v[idx == k]; w = w[np.isfinite(w)]
        if len(w): prof[k] = np.median(w) if stat == 'median' else w.mean()
    ok = np.isfinite(prof)
    if not ok.any(): return np.zeros(K)
    if not ok.all():
        x = np.arange(K); xi = np.concatenate([x[ok] - K, x[ok], x[ok] + K]); prof = np.interp(x, xi, np.tile(prof[ok], 3))
    return prof

def at_phase(prof, pos):
    """A cycle shape read at positions pos (in bins, any real number), straight lines between bin centres, round the cycle."""
    K = len(prof); q = pos - 0.5; i = np.floor(q).astype(np.int64); f = q - i; return prof[i % K] * (1 - f) + prof[(i + 1) % K] * f

def slow_fit(d, ok, T):
    """A movement slower than half the sound (seen less than twice): the best fit of a sine with its second and third harmonic, for
    periods from half the sound's length up to four times its length. Returns (period in frames, share explained, the fitted course)."""
    t = np.arange(T)[ok]; v = d[ok]; var = float(((v - v.mean()) ** 2).sum()) + 1e-12; fits = []
    for P in np.exp(np.linspace(np.log(T / 2.0), np.log(4.0 * T), 60)):
        w = 2 * np.pi / P; A = np.stack([np.ones(len(t))] + [f(h * w * t) for h in (1, 2, 3) for f in (np.cos, np.sin)], 1); c = np.linalg.lstsq(A, v, rcond=None)[0]; r = v - A @ c; fits.append((1 - float((r ** 2).sum()) / var, float(P), c))
    top = max(f[0] for f in fits)
    if top <= 0: return (0.0, None, None)
    return next(f for f in fits if f[0] >= top - 0.02)                       # the shortest period that fits (about) as well as the best

def find_pan(E):
    """Left and right moving against each other: measured from the difference of their levels, where the sound itself cancels."""
    T = E.shape[1]; d, D = pan_lane(E); ok = np.isfinite(d)
    if ok.sum() < 200: return None
    lo, hi = np.percentile(d[ok], [5, 95]); swing = float(hi - lo); mid = float(np.median(d[ok]))
    z, pw, _ = clip_lane(d)
    if pw <= 0 or swing < 0.5: return None
    Ps = np.exp(np.arange(np.log(60.0), np.log(T / 2.0), 0.004)); S = np.array([fold_stat(z, P, 24) for P in Ps]) / pw; k = int(np.argmax(S)); P = float(Ps[k]); share = float(S[k])
    cands = [(P, share)]
    for mlt in (2, 3, 4, 5, 6):                                                # a shorter period that explains (nearly) as much is the period
        j = int(np.argmin(np.abs(Ps - P / mlt)))
        if Ps[0] <= P / mlt:
            a = max(0, j - 3); jj = a + int(np.argmax(S[a:j + 4]))
            if S[jj] >= 0.9 * share: cands.append((float(Ps[jj]), float(S[jj])))
    P, share = min(cands); P = refine_period(z, P, 32, 0.01, 81); cyc = T / P; adj = (share - 1.0 / cyc) / (1.0 - 1.0 / cyc)      # few cycles explain much of anything
    r2s, Pslow, coef = slow_fit(d, ok, T); slow = bool(coef is not None and r2s > 0.7 and (adj < 0.6 or (cyc < 2.5 and r2s > adj + 0.05)))   # nothing that is seen twice explains the lane, a slower movement does
    if slow:
        P = Pslow; K = 64; w = 2 * np.pi / P; tt = (np.arange(K) + 0.5) / K * P
        prof = coef[0] + sum(coef[1 + 2 * i] * np.cos(h * w * tt) + coef[2 + 2 * i] * np.sin(h * w * tt) for i, h in enumerate((1, 2, 3))); share = adj = r2s; centre = float(0.5 * (prof.max() + prof.min()))
    else:
        K = int(np.clip(round(P / 8), 16, 64)); prof = fold_profile(d, P, K); centre = float(prof.mean())
    pos = (np.arange(T) % P) / P * K; fit = at_phase(prof, pos); res = d - fit; resid = float(np.nanmedian(np.abs(res[ok])) * 1.4826)
    sl = []                                                                    # does every band swing by the same amount?
    for b in range(D.shape[1]):
        okb = np.isfinite(D[:, b]) & ok
        if okb.sum() < 0.3 * ok.sum(): continue
        v = D[okb, b] - np.median(D[okb, b]); u = d[okb] - np.median(d[okb]); sl.append(float((v * u).sum() / ((u * u).sum() + 1e-12)))
    sl = np.array(sl) if sl else np.zeros(1); up, steep, fund, rise = shape_numbers(prof)
    return dict(P=float(P), K=K, prof=prof - centre, centre=centre, lane=d - centre, swing=swing, share=share, adj=float(adj), resid=resid, slow=slow, cycles=float(T / P),
                slope_med=float(np.median(sl)), slope_lo=float(np.percentile(sl, 25)), slope_hi=float(np.percentile(sl, 75)), bands=int(len(sl)), known=float(ok.mean()), fund=fund, steep=steep)

P_DEFAULT = 1.0
def fit_law(E, P, K, dprof, centre=0.0):
    """Which law do the two sides follow? The level of both sides together over the cycle (measured like a tremolo's shape) is set
    against what each law predicts from the measured left-right difference. Returns (p, how much better than the worst law)."""
    Et = E[0] + E[1]; L = cell_levels(np.stack([Et, Et]), P, K); c, r2, npairs = pairs_fold(L, K)
    if npairs == 0: return P_DEFAULT, 0.0
    a = 10 ** (centre / 10.0); wl, wr = a / (1 + a), 1 / (1 + a); best = (1e30, P_DEFAULT); errs = []
    for p in np.exp(np.linspace(np.log(0.25), np.log(6.0), 33)):
        gl, gr = law_gains(dprof, p); h = 10 * np.log10(wl * 10 ** (gl / 10) + wr * 10 ** (gr / 10)); e = float((((c - c.mean()) - (h - h.mean())) ** 2).mean()); errs.append(e)
        if e < best[0]: best = (e, float(p))
    return best[1], float(1.0 - best[0] / max(max(errs), 1e-12))

SHARE_MIN = 0.12; CYC_SPREAD = 9.0; STRUCK = 0.55                          # (the strict setting tried last: SHARE_MIN 0, CYC_SPREAD 0.9, STRUCK 0.35: see the report)
def first_filter(m):
    """Coarse: what does not pass this is never reported (and is not looked at more closely)."""
    return (m['P'] >= PMIN and m['depth'] >= 3.0 and m['corr_lo'] >= 0.6 and m['cyc_n'] >= 4 and m['halves'] >= 0.8                 # deep enough, seen often enough, even and odd cycles alike,
            and m['slope_hi'] - m['slope_lo'] <= 0.5 and abs(m['slope_med'] - 1.0) <= 0.5                                     # every band moving by about the same amount: what one gain on the whole sound does,
            and m['share'] >= SHARE_MIN and m['cyc_hi'] - m['cyc_lo'] <= CYC_SPREAD)                                          # explaining a fair share of what the level does, and about as far in every cycle

def struck(m):
    """A shape that jumps up and sinks back slowly is what struck and plucked notes do in every bar of music: never reported."""
    return bool(m['fall'] >= 0.15 and m['climb'] <= STRUCK * m['fall'] and m['fall'] * m['P'] >= 20.0)

def near1(v, e=0.02): return float(-np.log(max(1.0 - v, 0.0) + e))                # stretches the last stretch before 1: 0.9, 0.99, 0.999 lie evenly apart
RULE_NAMES = ['logdepth', 'corr_lo', 'corr_med', 'corr_min', 'slope_spread', 'slope_off', 'slope_range', 'cyc_off', 'cyc_spread', 'halves', 'r2c', 'share', 'excess', 'logcyc', 'steep', 'fund', 'silent', 'skipfrac', 'r2',
              'dis_med', 'dis_hi', 'flat', 'fall', 'climb', 'h_cyc_lo', 'h_cyc_hi', 'h_few', 'cyc60', 'h_slope_hi', 'h_slope_lo', 'h_halves', 'h_share', 'h_corr_lo']
def rule_features(m):
    """The measures as the rule sees them."""
    sp = m['slope_hi'] - m['slope_lo']; lg = lambda v, e: float(np.log(v + e))
    return [lg(min(max(m['depth'], 0.3), 60.0), 0.0), near1(m['corr_lo']), near1(m['corr_med']), near1(m['corr_min']), lg(min(sp, 3.0), 0.03), lg(min(abs(m['slope_med'] - 1.0), 2.0), 0.03), lg(min(max(m['slope_max'] - m['slope_min'], 0.0), 6.0), 0.05),
            lg(min(abs(m['cyc_med'] - 1.0), 2.0), 0.03), lg(min(max(m['cyc_hi'] - m['cyc_lo'], 0.0), 4.0), 0.03), near1(m['halves']), near1(max(m['r2c'], -1.0), 0.05), near1(m['share'], 0.05), near1(m['excess'], 0.05), lg(max(m['cyc_n'], 1), 0.0),
            float(m['steep']), near1(m['fund'], 0.05), float(m['silent']), float(m['skipfrac']), near1(max(m['r2'], -1.0), 0.05), lg(min(m['dis_med'], 2.0), 0.01), lg(min(m['dis_hi'], 2.0), 0.01), float(m['flat']), float(m['fall']), float(m['climb']),
            max(0.0, 0.7 - m['cyc_lo']), max(0.0, m['cyc_hi'] - 1.3), max(0.0, 12 - m['cyc_n']) / 8.0, min(m['cyc_n'], 60) / 60.0, max(0.0, m['slope_hi'] - 1.2), max(0.0, 0.8 - m['slope_lo']), max(0.0, 0.9 - m['halves']), max(0.0, 0.3 - m['share']),
            max(0.0, 0.9 - m['corr_lo'])]                                      # the last nine: how far outside the range true movements keep to

PAN_NAMES = ['logswing', 'adj', 'resid', 'slope_off', 'slope_spread', 'known', 'steep', 'fund', 'logcyc']
def pan_features(m):
    return [float(np.log(max(m['swing'], 0.1))), float(np.clip(m['adj'], -1.0, 1.0)), float(np.log(m['resid'] / max(m['swing'], 0.1) + 0.003)), float(np.log(min(abs(m['slope_med'] - 1.0), 2.0) + 0.01)),
            float(np.log(min(max(m['slope_hi'] - m['slope_lo'], 0.0), 3.0) + 0.01)), float(m['known']), float(m['steep']), near1(m['fund'], 0.05), float(np.log(max(m['cycles'], 0.25)))]

def load_rule():
    import json, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rule.json'); return json.load(open(p)) if os.path.exists(p) else {}
RULE = load_rule()                                                             # fitted weights and cuts (fit4.py, fit5.py); without the file: the hand rules below

def score_together(m):
    """How sure a shared movement is (the higher the surer); -inf: never reported."""
    if not first_filter(m) or struck(m): return -np.inf
    r = RULE.get('together')
    if r is None: return 1.0 if (m['corr_lo'] >= 0.8 and m['slope_hi'] - m['slope_lo'] <= 0.5 and 0.7 <= m['slope_med'] <= 1.4 and m['excess'] >= 0.08) else -1.0
    return float(np.dot(r['w'], rule_features(m)) + r['b'] - r['cut'])
def accept_together(m): return score_together(m) >= 0.0

PAN_ADJ_MIN = 0.2          # a pan's repeating shape explains at least this much of the lane over chance (every true pan of the fit data: 0.3 and more;
                           # echoes that fall on alternating sides swing the lane by tens of dB and repeat nothing)
SEEN_MIN = 1.25            # a movement from side to side seen less often than this (no part of it is seen coming back) may as well be the sound's
ONCE_CAP = -0.62           # own, something passing by: it scores no higher than this (a sureness of one half, which the default does not show)
def score_pan(m):
    if m['swing'] < 3.0 or m['known'] < 0.3 or m['slope_hi'] - m['slope_lo'] > 0.2 or m['adj'] < PAN_ADJ_MIN: return -np.inf       # a pan swings every band alike: 99 in 100 true pans keep within this, 1 in 15 clean sounds with a swing
    r = RULE.get('pan')
    if r is None: s = 1.0 if (0.75 <= m['slope_med'] <= 1.3 and m['slope_hi'] - m['slope_lo'] <= 0.4 and m['adj'] >= 0.6 and m['resid'] <= 0.25 * m['swing']) else -1.0
    else: s = float(np.dot(r['w'], pan_features(m)) + r['b'] - r['cut'])
    return min(s, ONCE_CAP) if m['cycles'] < SEEN_MIN else s
def accept_pan(m): return score_pan(m) >= 0.0

def choose_together(cands, rule=accept_together):
    """Of the candidates the rule lets through: the surest; the same movement seen at a shorter period (a whole fraction) wins if it passes too."""
    ok = [m for m in cands if rule(m)]
    if not ok: return None
    best = max(ok, key=score_together)
    for m in sorted(ok, key=lambda m: m['P']):
        r = best['P'] / m['P']
        if m['P'] < best['P'] * 0.75 and abs(r - round(r)) < 0.03 * round(r) and m['share'] >= 0.6 * best['share']: best = m; break
    return best

def find(x, info=None, rule_together=accept_together, rule_pan=accept_pan):
    """x: (n, 2) at 48 kHz. What was measured: dict(kind 'against' (left and right against each other: pan) or 'together' (tremolo, gate),
    period in seconds, rate in Hz, prof = the gain in dB per channel over one cycle (K, 2), top at 0 dB, ...) or None."""
    n = len(x); E = levels(x); T = E.shape[1]; pan = find_pan(E)
    if info is not None: info['pan'] = pan
    if pan is not None and rule_pan(pan):
        P, K = pan['P'], pan['K']; lane = None
        if pan['slow']:                                                   # less than two cycles seen: the course as it was measured, holes bridged
            d = pan['lane'].copy(); ok = np.isfinite(d); d = np.interp(np.arange(T), np.where(ok)[0], d[ok]); d = box(np.pad(d, 32, mode='edge'), 32)[32:-32]; p, lawfit = P_DEFAULT, 0.0
            gl, gr = law_gains(d, p); lane = np.stack([gl, gr], 1); pl, pr = law_gains(pan['prof'], p); prof = np.stack([pl, pr], 1); top = max(lane.max(), -1e9); lane = lane - top; prof = prof - top
        else:
            p, lawfit = fit_law(E, P, K, pan['prof'], pan['centre'])
            if lawfit < 0.8: p = P_DEFAULT                               # the level of both sides together does not say clearly: the usual law
            pl, pr = law_gains(pan['prof'], p); prof = np.stack([pl, pr], 1); prof -= max(top_of(prof[:, 0]), top_of(prof[:, 1]))
        return dict(kind='against', period=P * HOP / SR, rate=SR / HOP / P, K=K, prof=prof, lane=lane, n=n, law=p, lawfit=lawfit, depth=pan['swing'], measures={k: v for k, v in pan.items() if k not in ('prof', 'lane')})
    cands = candidates_together(E)
    if info is not None: info['together'] = cands
    best = choose_together(cands, rule_together)
    return together_result(x, E, best) if best is not None else None

def despike(g):
    """A gate that switches hard clicks at its edges: bands the sound leaves empty then show a short peak there, and the shared shape
    gets peaks (up to three bins wide) over its open level. Peaks narrower than five bins are cut off when that lowers the top by
    more than 3 dB and leaves a movement standing: most of the depth, or 6 dB of it (clicks can stand higher over the open level
    than the gate goes down; a gate that is open for a short moment only leaves nothing when its peak is cut, and is kept)."""
    import scipy.ndimage as ndi
    o = ndi.grey_opening(g, size=5, mode='wrap'); left = depth_of(o)
    return o if (top_of(g) - top_of(o) > 3.0 and (left >= 0.5 * depth_of(g) or left >= 6.0)) else g

def band_depths(L, K, g, start=0, skip=()):
    """Per band (of both channels): how far its level is down in the low quarter of the shape g against the high quarter (median over
    cycles); nothing sounding counts as 20 dB under the band's quietest moment."""
    import warnings
    n = (len(L) - start) // K; sk = set(skip); keep = [i for i in range(n) if i not in sk]
    if len(keep) < 2: return np.zeros(0)
    fl = np.isnan(L); low = np.nanmin(np.where(fl, np.inf, L), 0); low[~np.isfinite(low)] = 0.0; Lf = np.where(fl, low[None, :] - 20.0, L)
    C = Lf[start:start + n * K].reshape(n, K, -1)[keep]; S = (~fl)[start:start + n * K].reshape(n, K, -1)[keep]; gl = np.roll(g, -start); o = np.argsort(gl, kind='stable'); q = max(1, K // 4)
    d = C[:, o[-q:], :].mean(1) - C[:, o[:q], :].mean(1); d[~S[:, o[-q:], :].any(1)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); out = np.nanmedian(d, 0)
    return out[np.isfinite(out)]

def together_result(x, E, best, fine=True):
    """The found shared movement made ready for use: for a shape with steep steps the period and the steps are looked at finely first."""
    P, K, g, start, skip = best['P'], best['K'], despike(best['g']), best['start'], best['skip']; edges = []
    if fine and best['steep'] >= 0.25 and best['depth'] >= 4.0:
        ncyc = int(E.shape[1] / P) + 1; P2, ed = sharpen(x, P, K, g, start, skip, ncyc)
        if ed:
            L = cell_levels(E, P2, K); g2, r2, npairs = pairs_fold(L, K)
            if npairs > 0:
                g2 = despike(g2)
                st2, sk2 = 0, []
                if shape_numbers(g2)[1] >= 0.3 and depth_of(g2) >= 6.0: g2, st2, sk2 = gate_steps(L, K, g2)
                P3, ed3 = sharpen(x, P2, K, g2, st2, sk2, ncyc)
                if ed3: P, g, start, skip, edges = P3, g2, st2, sk2, ed3
    bd = band_depths(cell_levels(E, P, K), K, g, start, skip); trust = float(np.percentile(bd, 25) + 6.0) if len(bd) >= 4 else CAP_DB
    g = g - top_of(g); prof = np.stack([g, g], 1)
    return dict(kind='together', period=P * HOP / SR, rate=SR / HOP / P, K=K, prof=prof, lane=None, n=len(x), depth=depth_of(g), start=int(start), skip=[int(i) for i in skip], edges=edges, trust=trust,
                measures={k: v for k, v in best.items() if k != 'g'})

def gain_db(F, n):
    """The found gain in dB for samples 0 .. n-1 of a sound that starts where the analysed sound started: (n, 2)."""
    Pf = F['period'] * SR / HOP; K = F['K']; t = np.arange(n) / HOP; pos = (t % Pf) / Pf * K; g = np.stack([at_phase(F['prof'][:, ch], pos) for ch in range(2)], 1)
    for e in F.get('edges') or []:                                        # steep steps: their place and course as seen in the fine look
        Ps = Pf * HOP; k = e['k']; half = e['half']; rel = (np.arange(n) - e['off'] + half) % Ps; inside = rel < 2 * half
        if not inside.any(): continue
        r = rel[inside]; u = np.interp(r / DEC, np.arange(len(e['u'])), e['u']); taper = np.clip(np.minimum(r, 2 * half - r) / (0.25 * half), 0.0, 1.0)
        for ch in range(2):
            p = F['prof'][:, ch]; A = 0.5 * (p[(k - 2) % K] + p[(k - 1) % K]); B = 0.5 * (p[(k + 2) % K] + p[(k + 3) % K]); g[inside, ch] += taper * (A + u * (B - A) - g[inside, ch])
    if F.get('skip'):                                                     # steps the gate left out: closed all through
        low = np.sort(F['prof'], 0)[:max(1, K // 8)].mean(0); cyc = np.floor((t / Pf * K - F['start']) / K).astype(np.int64); g[np.isin(cyc, F['skip'])] = low
    if F.get('lane') is not None:                                         # the measured course while it lasts, the repeating shape after it
        L = F['lane']; m = min(n, (len(L) - 1) * HOP)
        for ch in range(2): g[:m, ch] = np.interp(t[:m], np.arange(len(L)), L[:, ch])
    return g

CAP_DB = 40.0; CLOSED = (-40.0, -55.0); GUARD = 48
def lift_db(g, trust=CAP_DB, low=None):
    """How far to lift (dB) to take the gain g (dB, (n, 2)) out again. Safeguards: never more than CAP_DB, nor more than `trust`
    (how deep the bands agree the movement goes); around every moment the gain is taken as the highest within 1 ms (an edge that is
    a fraction of a millisecond off then leaves a short dip, not a click); and where the gain is under CLOSED[0] the gate counts as
    closing: the lift fades and is nothing from CLOSED[1] down (what is left there is as likely noise as sound). For a shape with
    steep steps `low` (its low level) decides that for the whole movement, the steps included: half-way down a step that leads into a
    closed gate nothing is lifted either."""
    import scipy.ndimage as ndi
    gg = ndi.maximum_filter1d(g, 2 * GUARD + 1, axis=0, mode='nearest'); lift = np.minimum(-gg, min(CAP_DB, trust)); ref = gg if low is None else np.minimum(gg, low)
    return lift * np.clip((ref - CLOSED[1]) / (CLOSED[0] - CLOSED[1]), 0.0, 1.0) if low is None else lift * float(np.clip((low - CLOSED[1]) / (CLOSED[0] - CLOSED[1]), 0.0, 1.0))

def apply(x, F, s=-1.0):
    """s = -1: the found movement out of x (the sound it was found on). s = +1: onto x (a sound that starts where the analysed one started).
    Values in between and up to +2 scale the movement in dB (0: nothing, +2: twice as deep, -0.5: half taken out)."""
    g = gain_db(F, len(x))
    if s >= 0: return x * 10 ** (s * g / 20.0)
    low = float(np.sort(F['prof'][:, 0])[:max(1, F['K'] // 8)].mean()) if F.get('edges') else None
    return x * 10 ** (-s * lift_db(g, F.get('trust', CAP_DB), low) / 20.0)

# ---------------------------------------------------------------- sharp edges (a gate's steps, a sawtooth's drop)
DEC = 6                                                                    # the fine look at an edge: power in steps of 6 samples (0.125 ms)
def fine_power(x):
    """Power of both channels together in steps of DEC samples, each step set against the mean power of the second around it."""
    n = len(x) // DEC * DEC; q = (x[:n, 0] ** 2 + x[:n, 1] ** 2).reshape(-1, DEC).mean(1); m = SR // DEC; ref = box(q, m) + 1e-30 + 1e-9 * q.max(); return q / ref

def edges_of(g, depth):
    """Steep steps of a cycle shape: [(k, size in dB)], the step lies between bin k and bin k + 1; the largest rise and the largest fall, if steep."""
    K = len(g); d = np.diff(np.concatenate([g, g[:1]])); out = []
    for sgn in (1, -1):
        k = int(np.argmax(sgn * d)); tot = d[k] + (d[(k - 1) % K] if sgn * d[(k - 1) % K] > 0 else 0) + (d[(k + 1) % K] if sgn * d[(k + 1) % K] > 0 else 0)
        if sgn * tot >= max(4.0, 0.35 * depth) and sgn * d[k] >= 0.5 * sgn * tot: out.append((k, float(tot)))
    return out

def edge_fold(q, Ps, off, w, cyc):
    """Power around one edge, laid over all cycles: sum over cycles c of q at (c Ps + off) / DEC + (-w .. w)."""
    base = np.round((cyc * Ps + off) / DEC).astype(np.int64); tau = np.arange(-w, w + 1); idx = base[:, None] + tau[None, :]; ok = (idx >= 0) & (idx < len(q))
    v = np.where(ok, q[np.clip(idx, 0, len(q) - 1)], 0.0); return v.sum(0) / np.maximum(ok.sum(0), 1)

def contrast(F, m, sgn):
    """Where along F does the level step most (in the direction sgn), comparing the m points after with the m points before: (size in dB, place)."""
    cs = np.concatenate([[0.0], np.cumsum(F)]); n = len(F); t = np.arange(m, n - m); a = (cs[t + m] - cs[t]) / m; b = (cs[t] - cs[t - m]) / m
    c = sgn * 10 * np.log10(np.maximum(a, 1e-30) / np.maximum(b, 1e-30)); k = int(np.argmax(c)); return float(c[k]), int(t[k])

def sharpen(x, P, K, g, start, skip, ncyc):
    """A shape with steep steps, looked at finely: the period that makes the steps sharpest when all cycles are laid over each other, and
    each step's place and course in steps of 0.125 ms. Returns (period in frames, list of edges) with an edge = dict(off: its place in
    the cycle in samples, half: half width in samples of the stretch it replaces, u: its course there from 0 (level before) to 1 (after),
    in steps of DEC samples, before / after: bins of g that give the two levels) or (P, [])."""
    depth = depth_of(g); ed = edges_of(g, depth)
    if not ed: return P, []
    q = fine_power(x); Ps0 = P * HOP; binw = Ps0 / K; w = int(np.ceil((1.5 * binw + 0.004 * SR) / DEC)); m = 8
    sk = set(skip); cyc = np.array([c for c in range(-1, ncyc + 1)], dtype=np.float64)
    def use(k):                                                           # cycles to lay over each other for the edge after bin k (left-out steps do not have it)
        lc = np.floor((cyc * K + k + 1 - start) / K); return cyc[~np.isin(lc, list(sk))] if sk else cyc
    best = (-1.0, Ps0)
    for dlt in np.linspace(-4e-4, 4e-4, 41):
        Ps = Ps0 * (1 + dlt); tot = 0.0
        for k, size in ed: tot += contrast(edge_fold(q, Ps, (k + 1) / K * Ps - HOP / 2, w, use(k)), m, np.sign(size))[0]
        if tot > best[0]: best = (tot, Ps)
    Ps = best[1]; out = []
    for k, size in ed:
        sgn = np.sign(size); off0 = (k + 1) / K * Ps - HOP / 2; F = edge_fold(q, Ps, off0, w, use(k)); c, at = contrast(F, m, sgn)
        if c < 0.5 * abs(size): continue                                  # the fine look does not show the step: leave the coarse shape
        off = off0 + (at - w) * DEC; half = int(min(w, np.ceil((1.0 * binw + 0.003 * SR) / DEC))); F = edge_fold(q, Ps, off, half, use(k)); LdB = 10 * np.log10(np.maximum(F, 1e-30))
        e = max(2, half // 4); lo, hi = LdB[:e].mean(), LdB[-e:].mean()
        if abs(hi - lo) < 3.0: continue
        u = np.clip((LdB - lo) / (hi - lo), 0.0, 1.0); u = np.maximum.accumulate(u); u[:e // 2] = 0.0; u[-(e // 2):] = 1.0
        out.append(dict(off=float(off), half=half * DEC, u=u, k=int(k)))
    return Ps / HOP, out
