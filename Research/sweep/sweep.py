# Moving-filter engine, prototype: the flanger. A flanger is the sound plus a copy of itself whose delay moves. In the cepstrum
# of a short stretch of the sound that copy is a peak at its delay, as high as its gain; over the whole sound the peak draws a
# smooth line. The line (delay over time), the copy's gain and its feedback are measured from the one sound; the copy is then
# taken out by the exact inverse moving filter, or put on another sound by the same moving filter at the same times.
#   picture(x1)          the cepstrum picture of one channel (frames x lags), short lags from short windows, long lags from long ones
#   track(E, act, sign)  the smooth line through the picture that collects the most (dynamic programming), per sign of the copy
#   candidate(...)       one line followed up: sub-sample delay, a periodic curve (sine or triangle) where it fits, gain and feedback
#                        corrected until taking the copy out leaves nothing along the line, and the numbers the rule asks for
#   accept(f)            the rule: strong enough, reaching 1.3 ms, moving like a sweep (no jumps, not standing still like notes),
#                        gone after removal, and the level above 2 * SR / move fell by about what a copy of that gain adds
#   find(x)              what was measured, per channel: delay over time, gain, feedback, periodic curve; or None
#   apply(x, F, s, ...)  s = -1: out of the sound it was found on; s = +1: onto another sound; in between and up to +2: scaled;
#                        speed, sweep, feedback: the group's faders (see movement)
#   check(x, F)          the engine's own test of a result (is the line gone, did the level fall by what the copy added)
# Lags 8 .. 768 samples (0.17 .. 16 ms) at 48 kHz. Numbers behind the rule: stats.py, rule.py, final_stats.py; bench: ev.py.
import numpy as np, scipy.fft as sfft, scipy.ndimage as ndi
try:
    from numba import njit
except Exception:                                   # without numba everything still runs, slowly
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
SR = 48000; HOP = 256
SCALES = ((512, 8, 96), (1024, 96, 192), (2048, 192, 384), (4096, 384, 768))        # window, lags from .. to (samples)
QLO, QHI = SCALES[0][1], SCALES[-1][2]
F_LO, F_HI = 150.0, 16000.0                         # the part of the spectrum the picture is made from
FLOOR = 1e-5                                        # each frame's spectrum is held at this much of its top (100 dB under it)
KMAX = 10                                           # the line may move this many samples of delay from one frame to the next (39 ms/s)
LAM = 0.004                                         # what a move of one sample costs, squared

# ------------------------------------------------------------------ the picture
def _frames(x, n, hop=HOP):
    """Frames of one channel centred on multiples of hop (silence outside the sound): (T, n)."""
    T = len(x) // hop + 1; xp = np.concatenate([np.zeros(n // 2), x, np.zeros(n // 2 + hop)])
    return xp[np.arange(T)[:, None] * hop + np.arange(n)[None, :]]

_W = {}
def _weights(n):
    if n not in _W:
        f = np.arange(n // 2 + 1) * SR / n; w = np.zeros(n // 2 + 1); m = (f >= F_LO) & (f <= F_HI); w[m] = np.hanning(m.sum()); _W[n] = (w, np.hanning(n + 1)[:n])
    return _W[n]

def logspec(x1, n):
    """ln magnitude spectrum per frame, each held at FLOOR of its own top: (T, n/2+1); and the frames' energies."""
    w, win = _weights(n); F = _frames(x1, n) * win; S = np.abs(sfft.rfft(F, axis=1))
    return np.log(np.maximum(S, FLOOR * S.max(1, keepdims=True) + 1e-30)), (F ** 2).sum(1)

def _cep(L, n):
    """Cepstrum of ln spectra (T, n/2+1), scaled so that a copy of gain g at a delay of d samples gives g at lag d."""
    w = _weights(n)[0]; L = (L - (L * w).sum(1, keepdims=True) / w.sum()) * w
    return sfft.irfft(L, n, axis=1) * (n / w.sum())

def picture(x1, ref=None, need=None):
    """The cepstrum picture of one channel: (T, QHI), lag in samples, a frame every HOP samples. With ref (the same sound without
    the effect) the picture of the effect alone: the cepstrum of the difference of the two ln spectra.
    Also returns act (T,): frames that count (within 50 dB of the loud frames).
    need: lag ranges (from, to) that will be read; windows that serve none of them are left out (their lags stay 0, act is None)."""
    T = len(x1) // HOP + 1; E = np.zeros((T, QHI)); en = None
    for n, a, b in SCALES:
        if need is not None and not any(lo < b and hi >= a for lo, hi in need): continue
        L, e = logspec(x1, n)
        if ref is not None: L = L - logspec(ref, n)[0]
        E[:, a:b] = _cep(L, n)[:, a:b]
        if n == 1024: en = e
    if need is not None: return E, None
    top = np.percentile(en, 95); act = en > top * 1e-5
    return E, act

# ------------------------------------------------------------------ the line
@njit(cache=True)
def _dp(ev, kmax, lam):
    """The path q(t) that maximises the sum of ev[t, q(t)] minus lam * (step)^2, steps of at most kmax. Returns (path, total)."""
    T, Q = ev.shape; S = np.zeros((T, Q)); B = np.zeros((T, Q), np.int32)
    for q in range(Q): S[0, q] = ev[0, q]
    for t in range(1, T):
        for q in range(Q):
            best = -1e30; bk = 0
            for k in range(-kmax, kmax + 1):
                p = q + k
                if p < 0 or p >= Q: continue
                v = S[t - 1, p] - lam * k * k
                if v > best: best = v; bk = p
            S[t, q] = best + ev[t, q]; B[t, q] = bk
    path = np.zeros(T, np.int32); q = 0; best = -1e30
    for j in range(Q):
        if S[T - 1, j] > best: best = S[T - 1, j]; q = j
    for t in range(T - 1, -1, -1):
        path[t] = q; q = B[t, q]
    return path, best

def track(E, act, sign, cap=0.8):
    """The smooth line through the picture for copies of one sign (+1 or -1). Returns (path in samples, total collected)."""
    ev = np.clip(sign * E[:, QLO:], 0.0, cap) * act[:, None]
    path, tot = _dp(ev, KMAX, LAM); return path + QLO, float(tot)

def refine(E, act, path, sign, half=4, sig=4.0):
    """The line to a fraction of a sample: per frame the top of a parabola through the picture around the line, then a weighted
    smoothing over time (weights: how far the picture stands out there). Returns (d (T,) in samples, weight (T,), height (T,))."""
    T = len(path); d = path.astype(float).copy(); wt = np.zeros(T); hgt = np.zeros(T)
    for t in range(T):
        if not act[t]: continue
        a = max(QLO, path[t] - half); b = min(QHI - 1, path[t] + half); seg = sign * E[t, a:b + 1]; j = int(np.argmax(seg)); v = seg[j]
        if v <= 0: continue
        k = a + j; off = 0.0
        if 0 < j < len(seg) - 1:
            y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]; den = y0 - 2 * y1 + y2
            if den < 0: off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        d[t] = k + off; wt[t] = v; hgt[t] = v
    w = np.minimum(wt, 0.8) ** 2
    if w.sum() <= 0: return d, wt, hgt
    num = ndi.gaussian_filter1d(d * w, sig, mode='nearest'); den = ndi.gaussian_filter1d(w, sig, mode='nearest')
    ds = np.where(den > 1e-6, num / np.maximum(den, 1e-12), np.nan)
    ok = np.isfinite(ds)                                    # stretches without anything to see: straight lines between their ends
    if ok.any(): ds = np.interp(np.arange(T), np.where(ok)[0], ds[ok])
    else: ds = d
    return ds, wt, hgt

def along(E, d, k=1):
    """The picture read along k times the line (linear between lags): (T,). Outside the picture: nan."""
    T = len(d); q = k * d; i = np.floor(q).astype(int); f = q - i; ok = (i >= 1) & (i + 1 < E.shape[1]); i = np.clip(i, 0, E.shape[1] - 2)
    v = E[np.arange(T), i] * (1 - f) + E[np.arange(T), i + 1] * f; return np.where(ok, v, np.nan)

# ------------------------------------------------------------------ the moving filter
@njit(cache=True)
def _run(x, d, g, fb, inverse):
    """The moving comb on one channel. forward: y = x + g w, w(t) = u(t - d(t)), u = x + fb w. inverse: x = y - g w with the same w.
    d in samples per sample (at least 1), read with a cubic through four points (linear under 2 samples)."""
    n = len(x); u = np.zeros(n); out = np.zeros(n)
    for t in range(n):
        dd = d[t]
        if dd < 1.0: dd = 1.0
        p = t - dd; i = int(np.floor(p)); f = p - i; w = 0.0
        if i >= 1 and dd >= 2.0:
            a0 = u[i - 1]; a1 = u[i]; a2 = u[i + 1]; a3 = u[i + 2] if i + 2 < t else u[i + 1]
            w = a1 + 0.5 * f * (a2 - a0 + f * (2 * a0 - 5 * a1 + 4 * a2 - a3 + f * (3 * (a1 - a2) + a3 - a0)))
        elif i >= 0:
            w = u[i] * (1 - f) + u[i + 1] * f
        if inverse:
            v = x[t] - g * w; out[t] = v; u[t] = v + fb * w
        else:
            out[t] = x[t] + g * w; u[t] = x[t] + fb * w
    return out

def per_sample(d, n):
    """The line (one value per frame) for every sample of a sound of n samples."""
    return np.interp(np.arange(n) / HOP, np.arange(len(d)), d)

def from_peaks(c1, c2):
    """Gain and feedback of the copy from the picture's height at the delay (c1) and at twice the delay (c2):
    the filter (1 + r z^-d) / (1 - fb z^-d) has c1 = r + fb and c2 = (fb^2 - r^2) / 2; the copy's gain is g = r + fb."""
    if abs(c1) < 1e-6: return 0.0, 0.0
    dif = float(np.clip(2 * c2 / c1, -1.6, 1.6)); fb = 0.5 * (c1 + dif); r = 0.5 * (c1 - dif)
    fb = float(np.clip(fb, -0.9, 0.9)); r = float(np.clip(r, -0.97, 0.97)); return r + fb, fb

def movement(f, T, speed=1.0, sweep=1.0):
    """The found line for T frames, as the group's faders want it. speed: the line is read that many times as fast; where it
    ends (a longer sound, or speed above 1) a periodic movement carries on as its curve, a free one runs back and forth.
    sweep: the line's excursion around its middle is scaled (0: a comb that stands still at the middle, 1: as found)."""
    d = f['d']; n = len(d)
    if speed == 1.0 and sweep == 1.0 and T <= n: return d[:T]
    pos = np.arange(T) * speed; c = f.get('curve'); seen = f['weight'] > 0.05
    if c is not None:                               # the curve at any time, plus the line's own small departures while it lasts
        out = c['lo'] + (c['hi'] - c['lo']) * lfo_shape(c['rate'] * pos / FPS + c['phase'], c['kind'])
        out = out + np.where(pos <= n - 1, np.interp(pos, np.arange(n), d - lfo_curve(n, c)), 0.0); mid = 0.5 * (c['lo'] + c['hi'])
    else:
        m = 2.0 * (n - 1); q = np.mod(pos, m); q = np.where(q > n - 1, m - q, q); out = np.interp(q, np.arange(n), d)
        mid = float(np.median(d[seen])) if seen.any() else float(np.median(d))
    return np.clip(mid + sweep * (out - mid), 1.0, QHI - 1.0)

def render(x, F, remove, strength, speed=1.0, sweep=1.0, feedback=0.0):
    """x (n, 2) through what was found. remove: the found filter's inverse first (x is the sound it was found on); then the
    filter itself at `strength` times the copy's gain (0: nothing), its movement and feedback as the faders want them."""
    n = len(x); y = np.array(x, float, copy=True)
    for ch in range(2):
        f = F['ch'][ch]
        if f is None: continue
        g, fb = f['g'], f['fb']; v = np.ascontiguousarray(y[:, ch])
        if remove: v = _run(v, per_sample(f['d'], n), g, fb, True)
        if strength > 0: v = _run(v, per_sample(movement(f, n // HOP + 1, speed, sweep), n), float(np.clip(strength * g, -0.98, 0.98)), float(np.clip(fb + feedback, -0.9, 0.9)), False)
        y[:, ch] = v
    return y

def apply(x, F, s=-1.0, speed=1.0, sweep=1.0, feedback=0.0):
    """s = -1: the found effect out of x (the sound it was found on); -1 < s < 0: partly out (strength 1 + s is left);
    s = +1: onto x (another sound) as found; 0 < s <= 2: scaled; s = 0: x itself (on its own sound: with the faders moved,
    the copy taken out and put back as they say: render(x, F, True, 1, ...)).
    The faders (see movement): speed, sweep, and feedback (added to the found feedback) shape the copy that is put on."""
    if F is None or s == 0: return np.array(x, float, copy=True)
    return render(x, F, True, 1.0 + s, speed, sweep, feedback) if s < 0 else render(x, F, False, min(s, 2.0), speed, sweep, feedback)

# ------------------------------------------------------------------ the movement as a periodic curve
# delay(t) = lo + (hi - lo) * shape(rate * t + phase): shape 0 = sine, 1 = triangle, both 0 .. 1. Most flangers move like this;
# where the measured line fits such a curve, the curve replaces it: it is exact between the moments the sound shows the comb,
# it carries on where the sound is silent, and its rate and range are what the Speed and Sweep faders work on.
FPS = SR / HOP
RATES = np.geomspace(0.06, 4.0, 160); PHASES = np.arange(32) / 32.0
def lfo_shape(ph, kind): return 0.5 * (1 + np.sin(2 * np.pi * ph)) if kind == 0 else 1 - np.abs(2 * (ph - np.floor(ph)) - 1)
def lfo_curve(T, p): return p['lo'] + (p['hi'] - p['lo']) * lfo_shape(p['rate'] * np.arange(T) / FPS + p['phase'], p['kind'])
def collected(P, act, sign, d, cap=0.8):
    """What the picture holds along a line: the sum over the frames that count of the picture's value there (0 .. cap)."""
    return float((np.clip(sign * np.nan_to_num(along(P, d, 1)), 0.0, cap) * act).sum())
def lfo_fit(P, act, sign, d, w, keep=4):
    """The periodic curve nearest the measured line d (weights w): least squares over a grid of rates and phases (the error of a
    frame counts up to 8 samples, so stretches where the line went astray do not decide), the best few fitted once more on the
    frames they agree with, then moved step by step to where the picture holds the most. Returns (curve dict, collected) or None."""
    T = len(d); t = np.arange(T) / FPS; sw = float(w.sum())
    if sw <= 0: return None
    found = []
    for kind in (0, 1):
        for r in RATES:
            s = lfo_shape(r * t[None, :] + PHASES[:, None], kind); ms = (s * w).sum(1) / sw; md = float((d * w).sum() / sw)
            v = ((s - ms[:, None]) ** 2 * w).sum(1); b = ((s - ms[:, None]) * (d - md) * w).sum(1) / np.maximum(v, 1e-12); a = md - b * ms
            e = (np.minimum((d[None, :] - a[:, None] - b[:, None] * s) ** 2, 64.0) * w).sum(1) / sw; e[(v < 1e-6 * sw) | (b <= 0)] = 1e9
            j = int(np.argmin(e)); found.append((float(e[j]), kind, float(r), float(PHASES[j]), float(a[j]), float(a[j] + b[j])))
    found.sort(key=lambda z: z[0]); best = None
    for e, kind, r, ph, lo, hi in found[:keep]:
        p = dict(kind=kind, rate=r, phase=ph, lo=lo, hi=hi)
        s = lfo_shape(r * t + ph, kind); ok = (np.abs(d - lo - (hi - lo) * s) < 6.0) & (w > 0)             # once more on the frames that agree
        if ok.sum() >= 8:
            wi = w * ok; swi = wi.sum(); ms = (s * wi).sum() / swi; md = (d * wi).sum() / swi; v = ((s - ms) ** 2 * wi).sum()
            if v > 1e-9: b = ((s - ms) * (d - md) * wi).sum() / v; p['lo'] = float(md - b * ms); p['hi'] = float(md - b * ms + b)
        J = collected(P, act, sign, lfo_curve(T, p)); step = dict(rate=0.02 * p['rate'] + 0.002, phase=0.01, lo=1.0, hi=1.0)
        for it in range(40):                                                                                    # step by step: each of the four numbers up or down, steps halved when nothing helps
            moved = False
            for k in ('rate', 'phase', 'lo', 'hi'):
                for sgn in (1.0, -1.0):
                    q = dict(p); q[k] = p[k] + sgn * step[k]
                    if q['lo'] < QLO or q['hi'] > QHI - 2 or q['hi'] - q['lo'] < 4 or q['rate'] < 0.03: continue
                    Jq = collected(P, act, sign, lfo_curve(T, q))
                    if Jq > J + 1e-9: p, J, moved = q, Jq, True
            if not moved:
                step = {k: 0.5 * v for k, v in step.items()}
                if step['lo'] < 0.05: break
        if best is None or J > best[1]: best = (p, J)
    return best

# ------------------------------------------------------------------ measuring
MED = 31            # frames (165 ms): what stays put in the picture for longer than half of this counts as the sound's own (a held note)
def static_part(E): return ndi.median_filter(E, size=(MED, 1), mode='nearest')

@njit(cache=True)
def _hp(x, a):
    n = len(x); y = np.zeros(n); p1 = 0.0; q1 = 0.0; x1 = 0.0; z1 = 0.0
    for i in range(n):
        s = a * (p1 + x[i] - x1); x1 = x[i]; p1 = s
        t = a * (q1 + s - z1); z1 = s; q1 = t; y[i] = t
    return y
def high_pass(x, fc):
    """Two one-pole high-passes in a row at fc Hz."""
    return _hp(np.ascontiguousarray(x, dtype=np.float64), float(np.exp(-2 * np.pi * fc / SR)))

def level_change(x1, y1, act):
    """dB by which the level changed from x1 to y1 over the frames that count."""
    a = (_frames(x1, 1024) ** 2).sum(1)[act].sum(); b = (_frames(y1, 1024) ** 2).sum(1)[act].sum()
    return float(10 * np.log10(max(b, 1e-30) / max(a, 1e-30)))

def level_above(x1, y1, act, move):
    """The level change in the part of the spectrum where the sweep means something for the level: above 2 * SR / move Hz
    (move: how far the delay travels, in samples) every partial of the sound is carried across at least two whole teeth of
    the comb, so a copy of gain g adds 10 log10(1 + g^2 / (1 - fb^2)) dB there whatever the sound is."""
    fc = 2.0 * SR / max(move, 1.0); return level_change(high_pass(x1, fc), high_pass(y1, fc), act)

def candidate(x1, E, P, act, ref=None, rounds=3, quick=None, paths=1, S=None):
    """One line through the picture P (the whole picture E, or its moving part), followed up: delay over time, gain, feedback,
    and the numbers the decision is made from. quick: a test on the first numbers that says whether to go on (saves time).
    Of the lines tried (both signs of the copy, `paths` lines each) the one is kept along which the picture holds the most
    once the line is smoothed."""
    best = None
    for sign in (1, -1):
        Pk = P
        for k in range(paths):                              # the best line, and the best one after that line is blanked out
            path, tot = track(Pk, act, sign); d, wt, hgt = refine(P, act, path, sign); col = collected(P, act, sign, d)
            if best is None or col > best[0]: best = (col, sign, path, tot, d, wt, hgt)
            if k + 1 < paths:
                Pk = Pk.copy()
                for i in range(len(path)): Pk[i, max(0, path[i] - 6):path[i] + 7] = 0.0
    col, sign, path, tot, d, wt, hgt = best; n = len(x1); curve = None
    w0 = np.clip(hgt, 0.0, 0.8) * act
    if ref is None and (w0 > 0.05).sum() >= 12:
        got = lfo_fit(P, act, sign, d, w0 ** 2)
        if got is not None and got[1] >= 0.8 * collected(P, act, sign, d):                  # the curve holds (nearly) as much as the free line: the curve it is
            # ... but where the sound shows the line clearly and it lies near the curve, the line's own small departures are kept
            curve = got[0]; dc = lfo_curve(len(d), curve); dev = d - dc; ww = w0 ** 2 * (np.abs(dev) < 6.0)
            corr = ndi.gaussian_filter1d(ww * dev, 4.0, mode='nearest') / (ndi.gaussian_filter1d(ww, 4.0, mode='nearest') + 0.02)
            d = dc + np.clip(corr, -4.0, 4.0); hgt = np.clip(sign * np.nan_to_num(along(P, d, 1)), 0.0, None)
    w = np.clip(hgt, 0.0, 0.8) * act; sw = float(w.sum())
    if sw <= 1e-9 or (w > 0.05).sum() < 12: return None
    v1 = along(E, d, 1); v2 = np.nan_to_num(along(E, d, 2)); v1 = np.nan_to_num(v1)
    seen = w > 0.05; c1 = float(sign * np.percentile(sign * v1[seen], 75)); s1 = float((w * v1).sum()); c2 = float((w * v2).sum() / s1 * c1) if abs(s1) > 1e-9 else 0.0
    f = dict(d=d, sign=sign, total=tot, weight=w, act=act, path=path, curve=curve, periodic=float(curve is not None), c1_first=c1, move=float(d[seen].max() - d[seen].min()), seen=float(seen.sum() / max(act.sum(), 1)),
             slope=float(np.median(np.abs(np.diff(d))[seen[1:]])) * SR / HOP / 48.0, dmin=float(d[seen].min()), dmax=float(d[seen].max()))
    if quick is not None and not quick(f): f.update(g=0.0, fb=0.0, c1=c1, c2=c2, res=1.0, res2=1.0, level=0.0, expect=0.0, ok=False); return f
    # gain and feedback are corrected until taking the copy out leaves nothing along the line and along twice the line
    ds = per_sample(d, n); g, fb = from_peaks(c1, c2); res = res2 = 1.0; out = x1
    need = ((np.floor(d.min()) - 1, np.ceil(d.max()) + 2), (np.floor(2 * d.min()) - 1, np.ceil(2 * d.max()) + 2))      # the lags read below
    for k in range(rounds + 1):
        out = _run(np.ascontiguousarray(x1), ds, g, fb, True); E2, _ = picture(out, ref, need)
        res = float((w * np.nan_to_num(along(E2, d, 1))).sum() / s1); res2 = float((w * np.nan_to_num(along(E2, d, 2))).sum() / s1)
        if k == rounds: break
        if res < 0.9: c1 = c1 / (1 - res)                   # along the line the picture shows k(t) * c1 before and k(t) * (c1 - used) after
        else: c1 = c1 * 1.5
        c1 = float(np.clip(c1, -1.5, 1.5)); c2 = c2 + res2 * c1; g, fb = from_peaks(c1, c2)
    f.update(g=g, fb=fb, c1=c1, c2=c2, res=res, res2=res2, level=level_change(x1, out, act), expect=float(-10 * np.log10(1 + g * g / max(1 - fb * fb, 1e-3))))
    f['level_b'] = level_above(x1, out, act, f['move']); f.update(shape_numbers(E, S, act, f))
    return f

def shape_numbers(E, S, act, f):
    """What tells a moving copy from a melody's own line: (static) how much of what the picture holds along the line stands
    still there (a held note: all of it), (fast) how much of the tracked line's way is done in jumps of 4 samples a frame or
    more (a melody: most of it; a sweep: next to nothing)."""
    d, w, sign, path = f['d'], f['weight'], f['sign'], f['path']
    tot = float((w * np.clip(sign * np.nan_to_num(along(E, d)), 0.0, None)).sum())
    st = float((w * np.clip(sign * np.nan_to_num(along(S, d)), 0.0, None)).sum() / tot) if S is not None and tot > 1e-9 else 0.0
    k = np.abs(np.diff(path.astype(np.int64))); m = act[1:] & act[:-1]; tv = float(k[m].sum())
    return dict(static=st, fast=float(k[m][k[m] >= 4].sum() / tv) if tv > 0 else 0.0)

REACH = 64.0        # samples (1.3 ms): the line must reach at least this delay; below it the picture shows the sound's own moving colour (formants)
JUMPS = 0.30        # at most this much of the tracked line's way may be jumps
STILL = (0.6, 0.8)  # at most this much of the line may stand still (free movement, periodic movement)
def passes(res, level, expect):
    """The engine's test of a result: with the copy taken out the line is gone from the picture (res: what is left of it), and
    the level (above 2 * SR / move) fell by about what a copy of this gain and feedback adds (expect, dB): not less than a
    third of it (a comb pressed onto a sound that has none makes it louder), not more than 2.2 times (a note's own harmonics
    lose much more)."""
    if abs(res) > 0.3 or expect > -0.15: return False
    return 0.35 <= level / expect <= 2.2
def accept(f):
    """The rule for one channel: is this line a moving copy of the sound? (Numbers: see the report.)"""
    if f is None or 'level_b' not in f: return False
    if abs(f['g']) < 0.2 or f['move'] < 12 or f['dmax'] < REACH or f['seen'] < 0.15: return False     # too weak; standing still (the tone engine's case); too short; hardly seen
    if abs(f['fb']) > 0.75 or abs(f['c1_first']) > 1.0: return False                                  # rings like a pitch
    if f['fast'] > JUMPS or f['static'] > STILL[1 if f['periodic'] else 0]: return False              # moves like a melody, not like a sweep
    return passes(f['res'], f['level_b'], f['expect'])

def measure_channel(x1, ref=None, quick=None):
    """One channel: the candidates (a line through the whole picture, a line through its moving part), each followed up."""
    E, act = picture(x1, ref)
    if act.sum() < 16: return []
    out = []; S = static_part(E) if ref is None else None
    for kind in ('whole', 'moving') if ref is None else ('whole',):
        P = E if kind == 'whole' else E - S
        f = candidate(x1, E, P, act, ref, quick=quick, paths=2, S=S)
        if f is not None: f['from'] = kind; out.append(f)
    return out

def quick_test(f): return 0.15 <= abs(f['c1_first']) <= 1.0 and f['move'] >= 12 and f['seen'] >= 0.15 and f['dmax'] >= REACH       # what the rule asks anyway, as far as it shows before the follow-up

def find(x, ref=None, info=None, rule=accept):
    """x (n, 2). What was measured: dict(kind, ch = per channel the line d (samples of delay per frame of HOP samples), the
    copy's gain g and feedback fb, the periodic curve if the movement is one, and the decision's numbers), or None when nothing
    passes the rule. A sound with two different channels must show the copy on both, with the same sign.
    (ref: the same sound without the effect, which gives the picture of the effect alone: for tests only.)"""
    chans = []; same = np.array_equal(x[:, 0], x[:, 1]); e = (x ** 2).sum(0); quiet = [e[c] < 1e-4 * e.max() for c in range(2)]
    for c in range(2):
        if c == 1 and same: chans.append(chans[0]); continue
        if quiet[c]: chans.append(None); continue
        cands = measure_channel(np.ascontiguousarray(x[:, c]), None if ref is None else np.ascontiguousarray(ref[:, c]), quick_test)
        if info is not None: info.append(cands)
        good = [f for f in cands if rule(f)]
        chans.append(max(good, key=lambda f: abs(f['g']) * f['seen']) if good else None)
    for c in range(2):
        if chans[c] is None and not quiet[c]: return None                           # one side shows nothing: no report
    if chans[0] is None and chans[1] is None: return None
    if chans[0] is not None and chans[1] is not None and chans[0]['g'] * chans[1]['g'] < 0: return None
    f0 = next(f for f in chans if f is not None)
    return dict(kind='flanger', ch=chans, strength=float(np.mean([abs(f['g']) for f in chans if f is not None])), range_ms=(f0['dmin'] / 48.0, f0['dmax'] / 48.0),
                period_s=(1.0 / f0['curve']['rate'] if f0['curve'] else None), shape=(('sine', 'triangle')[f0['curve']['kind']] if f0['curve'] else 'free'))

def check(x, F):
    """The engine's own test of a result, on the sound it was found on: with the copy taken out, (1) how much of the line is left
    in the picture (res: 0 = gone, 1 = as before), (2) by how much the level fell above 2 * SR / move (level, dB) against what a
    copy of this gain and feedback adds (expect, dB). ok: both as they must be (passes) on every channel that has a copy.
    find() reports only what passes; the app can call this again on whatever it is about to play."""
    C = dict(ok=F is not None, res=[0.0, 0.0], level=[0.0, 0.0], expect=[0.0, 0.0])
    if F is None: return C
    for c in range(2):
        f = F['ch'][c]
        if f is None: continue
        x1 = np.ascontiguousarray(x[:, c]); E0, act = picture(x1); y1 = _run(x1, per_sample(f['d'], len(x1)), f['g'], f['fb'], True); E1, _ = picture(y1)
        s0 = float((f['weight'] * np.nan_to_num(along(E0, f['d']))).sum()); s1 = float((f['weight'] * np.nan_to_num(along(E1, f['d']))).sum())
        C['res'][c] = s1 / s0 if abs(s0) > 1e-9 else 1.0; C['level'][c] = level_above(x1, y1, act, f['move']); C['expect'][c] = float(-10 * np.log10(1 + f['g'] ** 2 / max(1 - f['fb'] ** 2, 1e-3)))
        C['ok'] = bool(C['ok'] and passes(C['res'][c], C['level'][c], C['expect'][c]))
    return C
