# The TRUE moving comb of a case, measured with the dry sound known: per frame the delay at which "dry plus a delayed copy of
# dry" explains the wet sound best (least squares in the time domain), a smooth line through those, then gain and feedback of
# the whole filter by least squares. Used as the truth the blind measurement is judged against, and for the ceiling.
import numpy as np, scipy.fft as sfft, scipy.ndimage as ndi
import sweep
from sweep import HOP, SR

def evidence(x, y, n=1024, qhi=sweep.QHI, hop=HOP):
    """ev[t, d]: the share of the wet frame's energy that a copy of the dry sound at delay d explains beyond the dry sound itself."""
    T = len(x) // hop + 1; pad = qhi + n; xp = np.concatenate([np.zeros(pad), x, np.zeros(n + hop)]); yp = np.concatenate([np.zeros(pad), y, np.zeros(n + hop)])
    m = sfft.next_fast_len(n + qhi, real=True); ev = np.zeros((T, qhi)); cs = np.concatenate([[0.0], np.cumsum(xp ** 2)])
    for t in range(T):
        s = pad + t * hop - n // 2; xf = xp[s:s + n]; yf = yp[s:s + n]; xl = xp[s - qhi:s + n]                # xl[qhi - d + i] = x(s + i - d)
        yy = yf @ yf; xx = xf @ xf
        if yy < 1e-12 or xx < 1e-12: continue
        XL = sfft.rfft(xl, m); cy = sfft.irfft(np.conj(sfft.rfft(yf, m)) * XL, m)[:qhi + 1][::-1]             # cy[d] = sum_i y(i) x(i - d)
        cx = sfft.irfft(np.conj(sfft.rfft(xf, m)) * XL, m)[:qhi + 1][::-1]                                    # cx[d] = sum_i x(i) x(i - d)
        d = np.arange(qhi); ee = cs[s + n - d] - cs[s - d]                                                    # energy of the delayed frame
        yx = yf @ xf; det = xx * ee - cx[:qhi] ** 2; ok = det > 1e-9 * xx * ee
        a = np.where(ok, (yx * ee - cy[:qhi] * cx[:qhi]) / np.where(ok, det, 1), 0); g = np.where(ok, (cy[:qhi] * xx - yx * cx[:qhi]) / np.where(ok, det, 1), 0)
        expl = a * yx + g * cy[:qhi]; e0 = yx * yx / xx; ev[t] = np.where(ok, np.maximum(expl - e0, 0) / yy, 0)
    return ev

def fit(x, y, lam=0.002):
    """One channel, dry x and wet y. Returns dict d (per frame, samples), a, g, fb (y = a x + g w, w = delayed (x + fb w)), err (dB)."""
    ev = evidence(x, y); ev[:, :sweep.QLO] = 0; en = np.array([0.0]); T = len(ev)
    F = sweep._frames(y, 1024); e = (F ** 2).sum(1); act = e > np.percentile(e, 95) * 1e-5
    path, tot = sweep._dp(np.ascontiguousarray(ev * act[:, None]), sweep.KMAX, lam)
    # sub-sample: parabola through the evidence at the path, weighted smoothing
    d = path.astype(float); w = np.zeros(T)
    for t in range(T):
        k = path[t]
        if not act[t] or k < 1 or k >= ev.shape[1] - 1: continue
        y0, y1, y2 = ev[t, k - 1], ev[t, k], ev[t, k + 1]; den = y0 - 2 * y1 + y2
        if den < 0: d[t] = k + float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        w[t] = y1
    if w.sum() > 0:
        num = ndi.gaussian_filter1d(d * w, 3.0, mode='nearest'); den = ndi.gaussian_filter1d(w, 3.0, mode='nearest'); ok = den > 1e-9
        if ok.any(): d = np.interp(np.arange(T), np.where(ok)[0], (num / np.maximum(den, 1e-30))[ok])
    ds = sweep.per_sample(d, len(x)); best = None
    for fb in np.arange(-0.9, 0.91, 0.05):
        wv = sweep._run(np.ascontiguousarray(x), ds, 1.0, float(fb), False) - x                              # the delayed part alone, with this feedback
        A = np.array([[x @ x, x @ wv], [x @ wv, wv @ wv]]); b = np.array([y @ x, y @ wv])
        try: a, g = np.linalg.solve(A, b)
        except Exception: continue
        err = float(((y - a * x - g * wv) ** 2).sum() / max((y ** 2).sum(), 1e-30))
        if best is None or err < best[0]: best = (err, a, g, float(fb))
    err, a, g, fb = best
    return dict(d=d, a=float(a), g=float(g / a) if abs(a) > 1e-6 else float(g), fb=fb, err=10 * np.log10(max(err, 1e-12)), weight=w, act=act, path=path, c1=float(g / a) if abs(a) > 1e-6 else 0.0, c2=0.0, sign=1 if g * a > 0 else -1)

def find(dry, wet):
    return dict(kind='flanger', ch=[fit(np.ascontiguousarray(dry[:, c]), np.ascontiguousarray(wet[:, c])) for c in range(2)])
