# The bend engine, prototype: with the meter's answer (per half-wave: by how many dB the box squeezed each level of the sound
# as it was, 14 levels under its peak), take the bend out of a sound, drive it further, or put it on another sound.
import numpy as np
import bfeat
GRID = bfeat.GRID; MAXSQ = 40.0; SPREAD = 6.0; SURE_AT = -2.0; SURE_TOP = 5.0; LOW_AT = -6.0; TOP_AT = 3.0; HARD_AT = 10.0; SHOWN = 0.65; BACK_PART = 0.75; BACK_LESS = 1.0
def smooth(xk, yk, x):
    """A smooth curve through rising points that never turns back between them (cubic pieces; the steepness at a point is the
    harmonic mean of the two straight pieces that meet there, at the two ends that of the one piece)."""
    xk = np.asarray(xk, float); yk = np.asarray(yk, float); h = np.diff(xk); d = np.diff(yk) / h; m = np.zeros(len(xk)); m[0] = d[0]; m[-1] = d[-1]
    for i in range(1, len(xk) - 1):
        if d[i - 1] * d[i] > 0: w1 = 2 * h[i] + h[i - 1]; w2 = h[i] + 2 * h[i - 1]; m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    x = np.asarray(x, float); i = np.clip(np.searchsorted(xk, x, 'right') - 1, 0, len(xk) - 2); t = (x - xk[i]) / h[i]
    return (2 * t ** 3 - 3 * t ** 2 + 1) * yk[i] + (t ** 3 - 2 * t ** 2 + t) * h[i] * m[i] + (-2 * t ** 3 + 3 * t ** 2) * yk[i + 1] + (t ** 3 - t ** 2) * h[i] * m[i + 1]
def tidy(S, K):
    """An answer made fit for use: no squeeze under zero, none growing faster than the level itself (1 - 1/K dB per dB:
    the way back then spreads a value at most K times as far apart), never falling towards the top, held at MAXSQ."""
    S = np.minimum(np.maximum(np.asarray(S, float), 0.0), MAXSQ); S = np.maximum.accumulate(S); r = S.copy(); r[0] = min(r[0], (1.0 - 1.0 / K) * (GRID[0] - (bfeat.SMALL - 3.0)))
    for i in range(1, len(r)): r[i] = min(r[i], r[i - 1] + (1.0 - 1.0 / K) * (GRID[i] - GRID[i - 1]))
    return r
class Curve:
    """One half-wave's curve both ways, on a table: inn (the value before the bend, in the scale where small values pass as they are) <-> out (the bent value, as in the sound)."""
    def __init__(s, S, vp, K=SPREAD):
        S = tidy(S, K); s.top = float(S[-1]); l = np.arange(-96.0, 12.0001, 1.0 / 16)
        lev = np.concatenate([[-60.0, bfeat.SMALL - 3.0], GRID, [12.0]]); sq = np.concatenate([[0.0, 0.0], S, [S[-1] + (1.0 - 1.0 / K) * 12.0]])        # above the peak: as flat as the cap allows
        q = smooth(lev, sq, l); s.vp = vp; s.xp = vp * 10 ** (s.top / 20); s.inn = s.xp * 10 ** (l / 20); s.out = s.inn * 10 ** (-q / 20)
        for i in range(1, len(s.out)): s.out[i] = max(s.out[i], s.out[i - 1] * (1.0 + 1e-9))        # the bent value never falls as the value before the bend rises
        # the flat zone: from the lowest level on from which the squeeze grows as fast as the cap allows all the way to the peak
        cap = np.diff(S) >= (1.0 - 1.0 / K) * np.diff(GRID) - 1e-6; j = len(GRID) - 1
        while j > 0 and cap[j - 1]: j -= 1
        s.lflat = float(GRID[j]) if j < len(GRID) - 1 else None; s.yflat = float(np.interp(s.lflat, l, s.out)) if s.lflat is not None else np.inf
    def back(s, v):
        """Bent value(s) v >= 0 -> the value before the bend. Under the table everything is a line, over it the table's last value."""
        v = np.asarray(v, float); return np.where(v < s.out[0], v, np.interp(v, s.out, s.inn))
    def fwd(s, x):
        """Value(s) x >= 0 before the bend -> bent."""
        x = np.asarray(x, float); return np.where(x < s.inn[0], x, np.interp(x, s.inn, s.out))
def sure(logit, top):
    """How sure a bend is, 0 .. 1: the weaker of the meter's verdict and the squeeze it reads at the peak. SHOWN (the Sureness
    slider where it stands at first) when the verdict is SURE_AT and the squeeze SURE_TOP dB."""
    e = min(logit - SURE_AT, top - SURE_TOP); return 1.0 / (1.0 + np.exp(-e * 0.5 - np.log(SHOWN / (1.0 - SHOWN))))
def shown(logit, top): return bool(logit >= SURE_AT and top >= SURE_TOP)            # at the default: the same as sure(...) >= SHOWN
def kept(logit, top): return bool(logit >= LOW_AT and top >= TOP_AT)                # at all: shows when the slider is turned down
def find(y, meter):
    """The meter's answer for one sound. meter: the app's model file (.bin) or the trainer's (.pt)."""
    import bmeter
    X, vp = bfeat.features(y); S, logit = bmeter.model(meter).run(X, 20 * np.log10(vp[0] / vp[1])); top = max(tidy(S[0], SPREAD)[-1], tidy(S[1], SPREAD)[-1])
    return dict(S=S, vp=vp, logit=logit, top=float(top), found=kept(logit, top), shown=shown(logit, top), sure=float(sure(logit, top)), kind='distortion' if top >= HARD_AT else 'saturation')
def halves(v, cp, cn, fn):
    return np.where(v >= 0, getattr(cp, fn)(np.maximum(v, 0)), -getattr(cn, fn)(np.maximum(-v, 0)))
def gamma(s, top):
    """The Amount as a drive: 0 = none of the found drive, 1 = as found, 2 = as far again in dB as the loudest values were squeezed."""
    return s * s if s <= 1.0 else 10 ** ((s - 1.0) * top / 20)
def careful(S):
    """The read squeeze as the way back uses it: three quarters of it, less 1 dB. Read too low, some of the bend stays in;
    read too high, values are spread apart that were never squeezed, which is worse than the bend."""
    return np.maximum(BACK_PART * np.minimum(np.maximum(np.asarray(S, float), 0.0), MAXSQ) - BACK_LESS, 0.0)
def own(y, S, vp, s, K=SPREAD, arch_db=0.0, plain=False):
    """The sound itself: s = 0 the bend walked back (as far as careful() goes; plain: all of what was read), 1 untouched,
    2 driven as far again. The sound's peak stays where it was."""
    if s == 1.0: return y
    if s < 1.0 and not plain: S = careful(S)
    cp, cn = Curve(S[0], vp[0], K), Curve(S[1], vp[1], K); x = halves(y, cp, cn, 'back'); g = gamma(s, max(cp.top, cn.top)); pk = bfeat.peak_of(y)
    if arch_db > 0.0:
        import arch
        for c in range(y.shape[1]):
            for pol, cv in ((1.0, cp), (-1.0, cn)):
                if cv.lflat is None: continue
                h = arch.arches(np.maximum(pol * x[:, c], 0.0), pol * y[:, c] >= cv.yflat, cv.xp * 10 ** ((cv.lflat + arch_db) / 20)); m = pol * y[:, c] >= cv.yflat; x[m, c] = pol * h[m]
    out = x if g <= 0.0 else halves(g * x, cp, cn, 'fwd'); return out * (pk / max(bfeat.peak_of(out), 1e-30))
def onto(b, S, vp, s, K=SPREAD):
    """Another sound through the found bend: its peak driven as far into the curve as the first sound's was."""
    if s <= 0.0: return b
    cp, cn = Curve(S[0], vp[0], K), Curve(S[1], vp[1], K); pb = bfeat.peak_of(b); g = gamma(s, max(cp.top, cn.top)); k = g * max(cp.xp, cn.xp) / max(pb, 1e-30)
    out = halves(k * b, cp, cn, 'fwd'); return out * (pb / max(bfeat.peak_of(out), 1e-30))
