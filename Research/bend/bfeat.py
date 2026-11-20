# The bend study: what the meter sees of a sound, one half-wave at a time (the values above zero, or those below), and the
# true answer for a training example: how far the box squeezed each level of the sound as it was.
import numpy as np
F = 2048; HOPF = 1024
AL = np.array([0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.965, 0.98, 0.99, 0.995, 0.999, 1.0])       # places in a frame's order of values (the half-wave, zeros where the other half is)
PLACE = np.array([1535, 1638, 1740, 1842, 1904, 1945, 1975, 2006, 2027, 2037, 2045, 2047])                    # AL as places in a sorted frame of 2048
LEV = np.array([-48.0, -36, -27, -20, -14, -9, -5, -2.5, -1, -0.3])                            # frames sorted by their peak against the half-wave's (dB): 10 classes, the last above -0.3
VAL = np.array([0.2, 0.4, 0.6, 0.8, 0.9])                                                     # values against the frame's peak: 6 classes
GRID = np.array([-30.0, -26, -22, -18, -15, -12, -9, -6, -4, -2, 0])                           # where the answer is given: levels of the sound as it was, in dB against its peak (of this half-wave)
SMALL = -34.0                                                                                  # values this far under the peak and further count as small: what the box does to them is its plain gain
NF = (len(AL) - 1) * len(LEV) + len(LEV) + (len(VAL) + 1) * len(LEV) + 4
def peak_of(y, pol=0.0):
    """A sound's peak without its few largest values (1 in 10000). pol = +1 or -1: of that half-wave alone (the other half counts as zeros)."""
    a = (np.abs(y) if pol == 0.0 else np.maximum(pol * y, 0.0)).ravel(); k = max(1, int(np.floor(len(a) * 1e-4 + 0.5))); return float(np.partition(a, len(a) - k)[len(a) - k])
def half(y, pol):
    """Features of one half-wave of y (n, ch): (NF,), and the half-wave's peak."""
    vref = max(peak_of(y, pol), 1e-9)
    q_acc = np.zeros((len(LEV), len(AL) - 1)); q_n = np.zeros(len(LEV)); s_acc = np.zeros((len(LEV), len(VAL) + 1)); s_n = np.zeros((len(LEV), len(VAL) + 1)); near = np.zeros(4); npos = 0
    for c in range(y.shape[1]):
        v = pol * y[:, c]; z = np.maximum(v, 0.0); n = (len(z) - F) // HOPF + 1
        fr = np.lib.stride_tricks.sliding_window_view(z, F)[::HOPF][:n]; srt = np.sort(fr, 1); Q = srt[:, PLACE]; p = Q[:, -1]
        lev = 20 * np.log10(np.maximum(p, 1e-12) / vref); ok = lev > LEV[0]; b = np.clip(np.searchsorted(LEV, lev, 'right') - 1, 0, len(LEV) - 1)
        rel = np.maximum(20 * np.log10(np.maximum(Q[:, :-1], 1e-12) / np.maximum(p[:, None], 1e-12)), -60.0)
        np.add.at(q_acc, b[ok], rel[ok]); np.add.at(q_n, b[ok], 1.0)
        d2 = np.empty_like(v); d2[1:-1] = (0.5 * (v[2:] - v[:-2])) ** 2; d2[0] = d2[-1] = 0.0                    # how steep the wave is at each value
        frd = np.lib.stride_tricks.sliding_window_view(d2, F)[::HOPF][:n]; pos = fr > 0; vb = np.searchsorted(VAL, fr / np.maximum(p[:, None], 1e-12), 'right')
        tot = (frd * pos).sum(1) / np.maximum(pos.sum(1), 1)
        for k in range(len(VAL) + 1):
            m = pos & (vb == k); cnt = m.sum(1); mean = (frd * m).sum(1) / np.maximum(cnt, 1); good = ok & (cnt >= 8) & (tot > 0)
            val = np.clip(10 * np.log10(np.maximum(mean, 1e-30) / np.maximum(tot, 1e-30)), -40.0, 20.0)
            np.add.at(s_acc[:, k], b[good], val[good]); np.add.at(s_n[:, k], b[good], 1.0)
        zz = z[z > 0]; npos += len(zz)
        for i, dbv in enumerate((0.1, 0.5, 1.0, 2.0)): near[i] += float((zz > vref * 10 ** (-dbv / 20)).sum())
    A = q_acc / np.maximum(q_n[:, None], 1.0); frac = q_n / max(q_n.sum(), 1.0); B = s_acc / np.maximum(s_n, 1.0)
    return np.concatenate([A.ravel() / 20.0, frac, B.ravel() / 10.0, np.log10(near / max(npos, 1) + 1e-6) / 3.0]).astype(np.float32), vref
def features(y):
    """Both half-waves: (2, NF), and their peaks (2,)."""
    a, va = half(y, 1.0); b, vb = half(y, -1.0); return np.stack([a, b]), np.array([va, vb])
def forward(u, y):
    """The box as one rising curve, from the sound as it was (u, any scale) and what the box made of it (y). Per half-wave:
    levels l (dB against u's peak of this half), the mean of what came out there against what comes out at zero (never under
    zero, rising), and u's peak. One fit over both halves, so the curve cannot fold at zero."""
    from sklearn.isotonic import IsotonicRegression
    uu = u.ravel(); yy = y.ravel(); steps = np.arange(-72.0, 6.01, 0.25); nb = len(steps) - 1; ups = [max(peak_of(u, 1.0), 1e-12), max(peak_of(u, -1.0), 1e-12)]
    code = np.full(len(uu), nb, int)                                                # 0 .. nb-1: the lower half from its top down; nb: the small values; nb+1 .. 2nb: the upper half
    for pol, up in ((1.0, ups[0]), (-1.0, ups[1])):
        i = np.searchsorted(up * 10 ** (steps / 20), pol * uu, 'right') - 1; ok = (i >= 0) & (i < nb); code[ok] = nb + 1 + i[ok] if pol > 0 else nb - 1 - i[ok]
        code[(i >= nb)] = -1
    ok = code >= 0; n = np.bincount(code[ok], None, 2 * nb + 1); sy = np.bincount(code[ok], yy[ok], 2 * nb + 1); su = np.bincount(code[ok], uu[ok], 2 * nb + 1); have = n >= 3
    mu = su[have] / n[have]; fit = IsotonicRegression(out_of_bounds='clip').fit(mu, sy[have] / n[have], sample_weight=n[have]).predict(mu); f0 = float(np.interp(0.0, mu, fit)); res = []
    for pol, up in ((1.0, ups[0]), (-1.0, ups[1])):
        m = pol * mu > up * 10 ** (steps[0] / 20); l = 20 * np.log10(pol * mu[m] / up); f = np.maximum(pol * (fit[m] - f0), 1e-12); o = np.argsort(l); res.append((l[o], f[o], up))
    return res
def answer(u, y):
    """The true answer for a training example. Per half-wave (2, len(GRID)): by how many dB a value of the sound as it was,
    at this level against its peak, comes out lower than the box's plain gain would have it. The plain gain is the box's gain
    for small values (SMALL dB under the peak and further), as a line through them."""
    out = np.zeros((2, len(GRID)), np.float32); fw = forward(u, y); uu = u.ravel(); yy = y.ravel()
    lim = max(fw[0][2], fw[1][2]) * 10 ** (SMALL / 20); sm = np.abs(uu) < lim
    if sm.sum() < 200: return out
    A = np.stack([np.maximum(uu[sm], 0.0), np.minimum(uu[sm], 0.0), np.ones(sm.sum())], 1); gp, gn, _ = np.linalg.lstsq(A, yy[sm], rcond=None)[0]       # one line per half through the small values, meeting at zero
    for i, ((l, f, up), g) in enumerate(zip(fw, (gp, gn))):
        if len(l) < 8 or g <= 0: continue
        out[i] = np.interp(GRID, l, 20 * np.log10(g * up) + l - 20 * np.log10(f))
    return np.clip(out, -40.0, 60.0).astype(np.float32)
