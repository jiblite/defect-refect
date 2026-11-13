# With the TRUE band-level answer: does spreading it inside each band by "where did the earlier sound ring" beat plain interpolation?
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, scipy.signal as ss, common as cm, truth, score, render, gbfeat
from multiprocessing import Pool
def refine(X, g_db, tau=0.2, floor_db=-30.0, mix=1.0):
    """Bin gains from band gains: inside a band the reverb power goes to the bins in proportion to what rang there before
    (leaky average of the bin's own past), the rest of each bin is direct sound."""
    E = np.abs(X[0]) ** 2 + np.abs(X[1]) ** 2; a = np.exp(-cm.DT / tau); r = ss.lfilter([0.0, 1.0 - a], [1.0, -a], E, axis=0) + 1e-12 * E.max()
    Eb = E @ cm.BM.T; rb = r @ cm.BM.T; rho = 1.0 - 10 ** (g_db / 10.0); Rb = rho * Eb                          # reverb power per band
    idx = np.searchsorted(cm.EDGES, np.arange(E.shape[1]), side='right') - 1; idx = np.clip(idx, 0, cm.NB - 1)       # the band of every bin
    Rf = Rb[:, idx] * r / np.maximum(rb[:, idx], 1e-30); Sf = np.maximum(E - Rf, 0.0)
    # keep each band's total direct power where the band-level answer put it
    Sb = Sf @ cm.BM.T; want = (1.0 - rho) * Eb; Sf = Sf * (want / np.maximum(Sb, 1e-30))[:, idx]; Sf = np.minimum(Sf, E)
    G = np.sqrt(Sf / np.maximum(E, 1e-30)); Gp = render.gains(g_db, 0.0); G = np.maximum(G, 10 ** (floor_db / 20)); out = Gp ** (1 - mix) * G ** mix
    out[:, :cm.EDGES[0]] = Gp[:, :cm.EDGES[0]]; out[:, cm.EDGES[-1]:] = Gp[:, cm.EDGES[-1]:]; return out
def one(d):
    c = truth.load(d); direct, rev = truth.parts(c); wet = c['wet']; ft = cm.features(wet, direct); sc0 = score.Scorer(direct); dw = sc0(wet); out = {}; n = len(wet)
    def sc(y): return 100.0 * (1.0 - sc0(y) / max(dw, 1e-9))
    gt, live = gbfeat.target(ft['LE'], ft['LS']); X = cm.fwd(wet); out['band gains, interpolated'] = sc(cm.inv(X * render.gains(gt)[None], n))
    for tau in (0.05, 0.2, 0.6): out['spread by past, tau %.2f' % tau] = sc(cm.inv(X * refine(X, gt, tau)[None], n))
    out['half and half, tau 0.2'] = sc(cm.inv(X * refine(X, gt, 0.2, mix=0.5)[None], n))
    Xr = cm.fwd(rev); Xd = cm.fwd(direct); Rt = np.abs(Xr[0]) ** 2 + np.abs(Xr[1]) ** 2; St = np.abs(Xd[0]) ** 2 + np.abs(Xd[1]) ** 2
    out['exact bins'] = sc(cm.inv(X * np.maximum(np.sqrt(St / np.maximum(St + Rt, 1e-30)), 10 ** -1.5)[None], n))
    return c['name'], out
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1])]
    with Pool(2) as p: rows = p.map(one, ds)
    keys = list(rows[0][1].keys()); a = np.array([[r[1][k] for k in keys] for r in rows])
    for i, k in enumerate(keys): print('%-28s median %5.1f  mean %5.1f' % (k, np.median(a[:, i]), np.mean(a[:, i])))
    print('%d cases, %.0f s' % (len(rows), time.time() - t0))
