# MOVEMENT faders (prototype): Loudness, Stereo, Tone, Spectral, and how a Pitch curve is put on. 48 kHz, (n, 2) or (n,).
# Every fader runs 0..2: 0 = flat, 1 = the sound as it is (nothing is done), 2 = the other sound's movement (two sounds) or
# the sound's own movement doubled (one sound). In between: straight lines in dB and cents.
#   find(x)                          what moves in a sound (see the parts below)
#   apply(x, own, other, faders)     x with the faders set; own = find(x); other = find(the other sound) or None (one sound)
# The other sound's movement goes on at the same times; past its end its last value holds.
# Order: Pitch (tape speed) first, then Spectral, Tone, Stereo, Loudness. After a pitch change the sound's own curves are
# carried along with it (each moment of the sound keeps its own values), the other sound's stay on the clock.
import os, sys, numpy as np, scipy.ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tone'))
import warp, tone, flow

SR = 48000
STEP = 480                  # the grid of pitch, stereo and tone: 10 ms
LSTEP = 240                 # the grid of loudness: 5 ms

def stereo(x):
    x = np.asarray(x, np.float64); return np.stack([x, x], 1) if x.ndim == 1 else x

def _hann(n): return 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)

def _frames(v, n, hop):
    """v (m,) in windows of n every hop, the first centred on sample 0: (T, n) with T = 1 + m // hop. Silence before the start
    and after the end."""
    T = 1 + len(v) // hop; vp = np.concatenate([np.zeros(n // 2), v, np.zeros(n + hop)])
    return vp[np.arange(T)[:, None] * hop + np.arange(n)[None, :]]

def _at(curve, T):
    """A curve of another sound at this sound's times: cut, or its last value held."""
    curve = np.asarray(curve); return curve[:T] if len(curve) >= T else np.concatenate([curve, np.repeat(curve[-1:], T - len(curve), 0)])

def _smooth(v, k):
    w = _hann(k); w /= w.sum(); return ndi.convolve1d(v, w, axis=0, mode='nearest')

# ------------------------------------------------------------------------------------------------ Loudness
BRIDGE = 8                  # frames either side (40 ms): a dip shorter than 80 ms between two peaks is bridged
LIFT_FULL, LIFT_NONE = -24.0, -50.0     # flat: parts this far under the sound's level are lifted all the way / not at all
def loud(x):
    """The level over time, dB every 5 ms: power in 10 ms windows; dips shorter than 80 ms are bridged (so the single pulses
    of a low engine, down to 13 a second, read as one even sound, while a hit keeps its edges); evened over 35 ms.
    And the sound's own level: the power average of all moments within 40 dB of the loudest."""
    p = (stereo(x) ** 2).mean(1); w = _hann(2 * LSTEP); P = _frames(p, 2 * LSTEP, LSTEP) @ (w / w.sum()); k = 2 * BRIDGE + 1
    E = _smooth(ndi.minimum_filter1d(ndi.maximum_filter1d(P, k, mode='nearest'), k, mode='nearest'), 7); top = E.max()
    if top <= 1e-20: return dict(db=np.full(len(E), -200.0), ref=-200.0)
    return dict(db=10 * np.log10(np.maximum(E, 1e-20)), ref=float(10 * np.log10(E[E >= top * 1e-4].mean())))

def lift(o):
    """The gain (dB) that evens a moment o dB off the sound's level; far under the level less and less (the floor is not pulled up)."""
    o = np.asarray(o, float); return np.where(o >= LIFT_FULL, -o, -LIFT_FULL * np.clip((o - LIFT_NONE) / (LIFT_FULL - LIFT_NONE), 0.0, 1.0))

def loud_gain(o, xo, v):
    """Gain in dB per 5 ms. o: the sound's own level against its own average; xo: the other sound's (None: one sound)."""
    if v <= 1.0: return (1.0 - v) * lift(o)
    if xo is None: return (v - 1.0) * np.clip(o, -40.0, 12.0)
    return (v - 1.0) * (lift(o) + np.maximum(xo, -80.0))

# ------------------------------------------------------------------------------------------------ Stereo
PAN_MAX = 24.0; WIDTH_MIN, WIDTH_MAX = -40.0, 6.0
def space(x):
    """Where the sound sits and how wide it is, every 10 ms. pan: right against left in dB (+ = right). width: side against mid in
    dB after the two channels were brought to the same level (a source panned to one side has no width of its own).
    pan0, width0: the averages (by power), the place the image stands still at."""
    x = np.asarray(x, np.float64); T = 1 + len(x) // STEP
    if x.ndim == 1 or np.abs(x[:, 0] - x[:, 1]).max() <= 1e-9 * max(np.abs(x).max(), 1e-30):
        return dict(mono=True, pan=np.zeros(T), width=np.full(T, WIDTH_MIN), pan0=0.0, width0=WIDTH_MIN, power=np.zeros(T))
    n = 5 * STEP; w = _hann(n); w /= w.sum(); l, r = x[:, 0], x[:, 1]
    pl = _smooth(_frames(l * l, n, STEP) @ w, 7); pr = _smooth(_frames(r * r, n, STEP) @ w, 7); c = _smooth(_frames(l * r, n, STEP) @ w, 7)
    tot = pl + pr; eps = 1e-10 * max(tot.max(), 1e-30); on = tot > 1e-4 * tot.max()
    pan = np.clip(10 * np.log10((pr + eps) / (pl + eps)), -PAN_MAX, PAN_MAX); a2 = 10 ** (pan / 20.0)        # left times a2, right over a2: both the same
    pm = (pl * a2 + pr / a2 + 2 * c) / 4; ps = (pl * a2 + pr / a2 - 2 * c) / 4
    width = np.clip(10 * np.log10((np.maximum(ps, 0) + eps) / (np.maximum(pm, 0) + eps)), WIDTH_MIN, WIDTH_MAX)
    if not on.any(): return dict(mono=False, pan=np.zeros(T), width=np.full(T, WIDTH_MIN), pan0=0.0, width0=WIDTH_MIN, power=tot)
    wt = tot * on; pan0 = float((wt * pan).sum() / wt.sum()); width0 = float((wt * width).sum() / wt.sum())
    pan = np.where(on, pan, pan0); width = np.where(on, width, width0)                                      # a silent moment: the average
    return dict(mono=False, pan=pan, width=width, pan0=pan0, width0=width0, power=tot)

def space_targets(pan, width, own, other, v):
    """The place and width the image is to have, from the sound's own (pan, width: at the moments of the sound now sounding)."""
    dp = pan - own['pan0']; dw = width - own['width0']
    if v <= 1.0 or other is None: tp = own['pan0'] + v * dp; tw = own['width0'] + v * dw
    else:
        T = len(pan); xp = _at(other['pan'], T) - other['pan0']; xw = (_at(other['width'], T) - other['width0']) if not other['mono'] else np.zeros(T)
        tp = own['pan0'] + (2.0 - v) * dp + (v - 1.0) * xp; tw = own['width0'] + (2.0 - v) * dw + (v - 1.0) * xw
    return np.clip(tp, -PAN_MAX, PAN_MAX), np.clip(tw, WIDTH_MIN, WIDTH_MAX)

def space_apply(y, pan, width, tp, tw):
    """y (n, 2) with the image moved from (pan, width) to (tp, tw), all per 10 ms; the power of every moment stays."""
    n = len(y); t = np.arange(n) / STEP; f = lambda v: np.interp(t, np.arange(len(v)), v)
    a = 10 ** (f(pan) / 40.0); l = y[:, 0] * a; r = y[:, 1] / a; m = 0.5 * (l + r); s = 0.5 * (l - r)
    k = np.where(width > WIDTH_MIN + 5.0, np.clip(tw - width, -24.0, 24.0), 0.0); s = s * 10 ** (f(k) / 20.0)        # nothing to widen in a sound without sides
    b = 10 ** (f(tp) / 40.0); o = np.stack([(m + s) / b, (m - s) * b], 1)
    N = 5 * STEP; w = _hann(N); w /= w.sum(); p0 = _frames((y ** 2).sum(1), N, STEP) @ w; p1 = _frames((o ** 2).sum(1), N, STEP) @ w
    g = np.sqrt((p0 + 1e-20) / (p1 + 1e-20)); return o * np.clip(f(g), 0.25, 4.0)[:, None]

# ------------------------------------------------------------------------------------------------ Tone
NT = 2048; NBAND = 18; TONE_MAX = 18.0
_f = np.fft.rfftfreq(NT, 1.0 / SR); _u = np.log2(np.maximum(_f, 1.0) / 44.0) * 2.0                         # in bands: centres 44 Hz x 2^(b/2), up to 15.9 kHz
_W = np.cos(0.5 * np.pi * np.clip(_u[None, :] - np.arange(NBAND)[:, None], -1.0, 1.0)) ** 2
_W[0, _u < 0] = 1.0; _W[-1, _u > NBAND - 1] = 1.0; _W[:, 0] = 0.0                                          # the ends run on flat; no DC
def whole_frames(found, T):
    """For each of T frames (NT long, every STEP, the first centred on sample 0) the frame whose reading counts. A frame that
    lies across a cut (found: flow.cuts, the file's ends among them) holds a sound that stops inside it, and that reads as a
    click: 18 dB and more in the bands where the sound has least, over the 30 ms either side of every cut, however steady the
    sound is. Such a frame takes the reading of the nearest frame that lies whole on the sounding side of the cut, so the
    balance stands still there; -1 where there is no such frame (a piece of sound shorter than a frame)."""
    src = np.arange(T); zones = []
    for p, d in found:
        lo = (p - NT // 2) // STEP; hi = -((-(p + NT // 2)) // STEP)            # the last frame that ends at or before the cut, the first that starts at or after it
        zones.append((lo + 1, hi - 1, hi if d > 0 else lo))
    spoiled = np.zeros(T, bool)
    for a, b, h in zones: spoiled[max(0, a):max(0, b + 1)] = True
    for a, b, h in zones: src[max(0, a):max(0, b + 1)] = h if (0 <= h < T and not spoiled[h]) else -1
    return src

def tone_move(x):
    """How the balance of 18 bands (an octave wide, every half octave) moves, dB per band every 10 ms: each band's level against
    the moment's average over the bands that carry the sound (weights w: 1 for a band within 30 dB of the sound's strongest
    band, 0 from 50 dB under), and that against its own average over time. Level-free, and it adds up: a gain per band changes
    it by exactly that gain (less the gains' weighted average). A quiet moment (60 to 40 dB under the loudest) counts less.
    Beside a hard cut the balance stands still (whole_frames)."""
    x = stereo(x); w = _hann(NT); T = 1 + len(x) // STEP; src = whole_frames(flow.cuts(0.5 * (x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]), True), T); own = src == np.arange(T)
    P = sum(np.abs(np.fft.rfft(_frames(x[:, c], NT, STEP) * w, axis=1)) ** 2 for c in range(2))
    E = (P @ _W.T) * own[:, None]; tot = E.sum(1); top = tot.max()
    if top <= 1e-20: return dict(m=np.zeros((T, NBAND)), w=np.zeros(NBAND))
    on = tot > 1e-4 * top; lt = 10 * np.log10(np.maximum(E[on].mean(0), 1e-30)); wb = np.clip((lt - (lt.max() - 50.0)) / 20.0, 0.0, 1.0)
    L = 10 * np.log10(np.maximum(E, 1e-9 * top)); m = L - (L @ wb / wb.sum())[:, None]; wt = tot * on; m = m - (wt[:, None] * m).sum(0) / wt.sum()
    m = m * np.clip((10 * np.log10(np.maximum(tot, 1e-30) / top) + 60.0) / 20.0, 0.0, 1.0)[:, None]         # a silent frame says nothing, also not to its neighbours
    m = np.where((src >= 0)[:, None], m[np.maximum(src, 0)], 0.0)                                                # across a cut: the reading of the whole frame beside it
    m = ndi.convolve1d(m, np.array([0.25, 0.5, 0.25]), axis=0, mode='nearest')
    return dict(m=np.clip(m, -TONE_MAX, TONE_MAX), w=wb)

def tone_gain(m, w, xm, xw, v):
    """Gain in dB per band and 10 ms. m, w: the sound's own movement and band weights; xm, xw: the other sound's (None: one sound).
    A band the sound hardly has is left alone; a band the other sound hardly has counts as standing still."""
    g = (v - 1.0) * m if (v <= 1.0 or xm is None) else (v - 1.0) * (_at(xm, len(m)) * xw[None, :] - m)
    return np.clip(g, -TONE_MAX, TONE_MAX) * w[None, :]

def tone_apply(y, g):
    """y (n, 2) with the band gains g (T, 18) dB per 10 ms; every moment keeps its power."""
    n = len(y); hop = NT // 4; w = np.sqrt(_hann(NT)); J = 1 + (n + NT) // hop; yp = np.vstack([np.zeros((NT, 2)), y, np.zeros((NT + hop, 2))])
    Wn = _W / np.maximum(_W.sum(0), 1e-9)[None, :]; out = np.zeros_like(yp); tj = (np.arange(J) * hop - NT // 2) / STEP           # frame centres on the 10 ms grid
    for a in range(0, J, 512):
        jj = np.arange(a, min(J, a + 512)); idx = jj[:, None] * hop + np.arange(NT)[None, :]
        X = np.stack([np.fft.rfft(yp[idx, c] * w, axis=1) for c in range(2)]); tt = np.clip(tj[jj], 0, len(g) - 1)
        i0 = np.floor(tt).astype(int); i1 = np.minimum(i0 + 1, len(g) - 1); fr = (tt - i0)[:, None]; gb = g[i0] * (1 - fr) + g[i1] * fr
        G = 10 ** ((gb @ Wn) / 20.0); p0 = (np.abs(X) ** 2).sum((0, 2)); Y = X * G[None]; p1 = (np.abs(Y) ** 2).sum((0, 2))
        Y = Y * np.sqrt((p0 + 1e-30) / (p1 + 1e-30))[None, :, None]
        for c in range(2): np.add.at(out[:, c], idx, np.fft.irfft(Y[c], NT, axis=1) * w)
    return out[NT:NT + n] / 2.0

# ------------------------------------------------------------------------------------------------ Spectral
_fb = np.maximum(tone.FB, tone.FB[1]); NEUTRAL = -10 * np.log10(np.clip(_fb, 20.0, 20000.0))                 # pink: the same power in every octave
COLOUR_OCT = 1.0 / 6.0; COLOUR_HZ = 40.0     # a colour: the long-term power evened over a third of an octave, and over no less than about 160 Hz
def _even(v, sigma_oct=COLOUR_OCT, sigma_hz=COLOUR_HZ):
    """A curve on the bins evened by a bell sigma_oct octaves wide (at least sigma_hz): three passes of a running mean."""
    n = len(v); k = np.arange(n, dtype=float); sig = np.maximum(sigma_oct * np.log(2.0) * k, sigma_hz / (tone.FB[1] - tone.FB[0])); h = 0.5 * (np.sqrt(4 * sig * sig + 1) - 1); pos = np.arange(n + 1) - 0.5
    for _ in range(3):
        cs = np.concatenate([[0.0], np.cumsum(v)]); a = np.clip(k - h - 0.5, -0.5, n - 0.5); b = np.clip(k + h + 0.5, -0.5, n - 0.5)
        v = (np.interp(b, pos, cs) - np.interp(a, pos, cs)) / np.maximum(b - a, 1e-9)
    return v

def colour(S):
    """A long-term spectrum (dB on tone.FB) in broad strokes: its power evened over a third of an octave (not its dB: the gaps
    between the partials of a sound with few of them do not count as dark)."""
    return 10 * np.log10(np.maximum(_even(10 ** ((S - S.max()) / 10.0)), 1e-30)) + S.max()

def spec_curve(own, other, v):
    """The fixed curve (dB on tone.FB). own, other: long-term spectra (tone.long_term_db). 0: the sound's colour brought to the
    neutral one; 2: to the other sound's colour, or (one sound) its own colour once more. Taken where the sound has something
    (within 60 dB of its strongest region; outside, the nearest value holds); at most +-30 dB, bent in softly as every tone
    curve of the app; the sound keeps its long-term power."""
    z = np.zeros(len(tone.FB))
    if own.max() < -250 or v == 1.0: return z
    co = colour(own); two = v > 1.0 and other is not None
    if two and other.max() < -250: return z
    ct = colour(other) if two else NEUTRAL; ok = (co > co.max() - 60) & (tone.FB >= 20)
    if two: ok &= ct > ct.max() - 60
    i = np.flatnonzero(ok)
    if not len(i): return z
    c = np.interp(np.arange(len(z)), i, (ct - co)[i]); c = _even(c); c = np.clip(c - np.median(c[i]), -30.0, 30.0)
    c = c * ((1.0 - v) if v <= 1.0 else (v - 1.0) if two else -(v - 1.0)); p = 10 ** ((own - own.max()) / 10)
    for _ in range(3): c = c - 10 * np.log10(max((p * 10 ** (tone._soft_cap(c) / 10)).sum() / p.sum(), 1e-12))
    return c

# ------------------------------------------------------------------------------------------------ Pitch
def pitch_positions(o, xo, v, n):
    """Where in the sound each moment of the result is read (tape speed). o: the sound's own pitch movement, cents per 10 ms
    (0 where none is followed); xo: the other sound's or None. The length stays: the sound keeps its average pitch."""
    t = np.arange(n) / STEP; f = lambda c: np.interp(t, np.arange(len(c)), c)
    a, b = (v - 1.0, 0.0) if (v <= 1.0 or xo is None) else (1.0 - v, v - 1.0)                              # the change: a x own (at the sound's moments) + b x other (on the clock)
    A = np.concatenate([[0.0], np.cumsum(2.0 ** (-a * f(o) / 1200.0))]); B = np.concatenate([[0.0], np.cumsum(2.0 ** (b * f(_at(xo, len(o))) / 1200.0))]) if b else np.arange(n + 1.0)
    return np.interp(B[:n] * (A[-1] / B[-1]), A, np.arange(n + 1.0))

# ------------------------------------------------------------------------------------------------ all of it
def find(x, pitch=None):
    """pitch: a function x_mono -> (cents per 10 ms, held) (the reader); None leaves the pitch part out."""
    x = np.asarray(x, np.float64); F = dict(n=len(x), loud=loud(x), space=space(x), tone=tone_move(x), spec=tone.long_term_db(x))
    if pitch is not None:
        c, held = pitch(stereo(x).mean(1))[:2]; F['pitch'] = dict(cents=c, held=held)
    return F

def apply(x, own, other, faders):
    """faders: dict with any of pitch, loudness, tone, stereo, spectral (0..2; missing = 1)."""
    v = lambda k: float(faders.get(k, 1.0)); y = stereo(x).copy(); n = len(y); pos = None
    if v('pitch') != 1.0 and 'pitch' in own and (other is None or 'pitch' in other or v('pitch') < 1.0):
        pos = pitch_positions(own['pitch']['cents'], other['pitch']['cents'] if (other is not None and v('pitch') > 1.0) else None, v('pitch'), n); y = warp.read_at(y, pos)
    def mine(curve, step):                  # the sound's own curve at the moments now sounding
        if pos is None: return curve
        k = np.minimum(np.arange(len(curve)) * step, n - 1); return curve[np.clip(np.round(pos[k] / step).astype(int), 0, len(curve) - 1)]
    if v('spectral') != 1.0:
        y = tone.put_curve(y, tone.FB, spec_curve(own['spec'], other['spec'] if other is not None else None, v('spectral')), 1.0)
    if v('tone') != 1.0: y = tone_apply(y, tone_gain(mine(own['tone']['m'], STEP), own['tone']['w'], other['tone']['m'] if other is not None else None, other['tone']['w'] if other is not None else None, v('tone')))
    if v('stereo') != 1.0:
        S = own['space']; pan, width = mine(S['pan'], STEP), mine(S['width'], STEP); tp, tw = space_targets(pan, width, S, other['space'] if other is not None else None, v('stereo')); y = space_apply(y, pan, width, tp, tw)
    if v('loudness') != 1.0:
        o = mine(own['loud']['db'], LSTEP) - own['loud']['ref']; xo = (_at(other['loud']['db'], len(o)) - other['loud']['ref']) if other is not None else None
        y = y * 10 ** (np.interp(np.arange(n) / LSTEP, np.arange(len(o)), loud_gain(o, xo, v('loudness'))) / 20.0)[:, None]
    return y
