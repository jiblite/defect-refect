# De-reverb variants side by side on the test cases (true room numbers unless a variant says otherwise).
import sys, json, glob, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, numba, scipy.signal as ss, room, truth, dereverb as dv, score
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/reverb/'; corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
def bandpow(X): return (np.abs(X[0]) ** 2 + np.abs(X[1]) ** 2)
@numba.njit(cache=True)
def dd_gain(E, R, alpha, over, gmin):
    """Decision-directed Wiener gain: the direct-to-reverb ratio of a bin leans on what was let through a frame earlier, so
    the gain moves smoothly through a tail instead of flickering with every bin's ups and downs."""
    T, F = E.shape; G = np.ones((T, F)); prev = np.zeros(F)
    for t in range(T):
        for f in range(F):
            r = over * R[t, f] + 1e-30; post = E[t, f] / r - 1.0
            if post < 0.0: post = 0.0
            xi = alpha * prev[f] / r + (1.0 - alpha) * post; g = xi / (1.0 + xi)
            if g < gmin: g = gmin
            G[t, f] = g; prev[f] = g * g * E[t, f]
    return G
def unshared_bins(X, tavg=0.128, favg=2):
    pl = np.abs(X[0]) ** 2; pr = np.abs(X[1]) ** 2; c = X[0] * np.conj(X[1]); k = max(1, int(round(tavg / dv.DT))); kt = np.ones((k, 1)) / k; kf = np.ones((1, 2 * favg + 1)) / (2 * favg + 1)
    sm = lambda a: ss.fftconvolve(ss.fftconvolve(a, kt, 'same'), kf, 'same')
    pls, prs, cs = sm(pl), sm(pr), sm(c.real) + 1j * sm(c.imag)
    return np.maximum((pls + prs) - np.sqrt(np.maximum((pls - prs) ** 2 + 4 * np.abs(cs) ** 2, 0.0)), 0.0), pls + prs
def one(d):
    c = truth.load(d); t = truth.room_truth(c); wet = c['wet']; direct, rev = truth.parts(c)
    tau = np.where(np.isfinite(t['tau']), t['tau'], np.nanmedian(t['tau'])); kap = np.where(np.isfinite(t['kappa']), t['kappa'], np.nanmedian(t['kappa']))
    if not np.isfinite(tau).all(): return None
    sc0 = score.Scorer(direct); dw = sc0(wet); out = {}; n = len(wet)
    def sc(y): return 100.0 * (1.0 - sc0(y) / max(dw, 1e-9))
    X = dv.fwd(wet); E = bandpow(X); Es = dv.smooth_tf(E); Rt = bandpow(dv.fwd(rev)); St = bandpow(dv.fwd(direct))
    out['exact bins'] = sc(dv.inv(X * np.sqrt(St / np.maximum(St + Rt, 1e-30))[None], n))
    Rm = dv.reverb_power(Es, tau, kap / tau)
    for over in (1.0, 1.5, 2.0, 3.0): out['model x%.1f' % over] = sc(dv.inv(X * dv.gain_from(Es, Rm, 0.0, -30.0, over)[None], n))
    for alpha in (0.85, 0.95):
        for over in (1.0, 2.0, 3.0): out['model dd%.2f x%.1f' % (alpha, over)] = sc(dv.inv(X * dd_gain(E, Rm, alpha, over, 10 ** -1.5)[None], n))
    # stereo: per bin, the power the channels do not share, scaled by how much of the true reverb part is unshared (its width)
    D, P = unshared_bins(X); Dr, Pr = unshared_bins(dv.fwd(rev)); u = max(Dr.sum() / max(Pr.sum(), 1e-30), 1e-3); Rst = np.minimum(D / u, P)
    out['unshared (true width)'] = sc(dv.inv(X * dv.gain_from(P, Rst, 0.0, -30.0, 1.0)[None], n))
    out['unshared x1.5'] = sc(dv.inv(X * dv.gain_from(P, Rst, 0.0, -30.0, 1.5)[None], n))
    out['width'] = 10 * np.log10(u)
    return c['name'], corp[c['meta']['A']]['kind'], float(np.nanmean(t['drr_db'][(room.FC > 450) & (room.FC < 7000)])), dw, out
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob(S + '*/'))
    if len(sys.argv) > 1 and sys.argv[1].isdigit(): ds = ds[::int(sys.argv[1])]
    with Pool(4) as p: rows = [r for r in p.map(one, ds) if r]
    keys = list(rows[0][4].keys())
    if '-v' in sys.argv:
        for r in rows: print('%-24s %-16s DRR %6.1f  wet is %4.1f dB from dry | ' % r[:4] + ' | '.join('%5.1f' % (r[4][k]) for k in keys))
    a = np.array([[r[2], r[3]] + [r[4][k] for k in keys] for r in rows])
    for lab, sel in [('all', np.ones(len(a), bool)), ('light (DRR >= 5)', a[:, 0] >= 5), ('middle (-2..5)', (a[:, 0] < 5) & (a[:, 0] >= -2)), ('wet (< -2)', a[:, 0] < -2)]:
        print('%-18s n=%3d' % (lab, sel.sum()))
        for i, k in enumerate(keys): print('    %-26s median %5.1f  mean %5.1f  worse than untouched in %2.0f%%' % (k, np.median(a[sel, 2 + i]), np.mean(a[sel, 2 + i]), 100 * np.mean(a[sel, 2 + i] < 0)))
    print('%.0f s' % (time.time() - t0))
