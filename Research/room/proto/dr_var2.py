# De-reverb variants, round 2 (true room numbers): where the "a bit more room than measured" goes, band-level against bin-level tracking.
import sys, json, glob, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, numba, scipy.signal as ss, room, truth, dereverb as dv, score
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/reverb/'; corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
def bandpow(X): return (np.abs(X[0]) ** 2 + np.abs(X[1]) ** 2)
BM = room.band_matrix(dv.NF); LF = np.log(np.maximum(np.fft.rfftfreq(dv.NF, 1 / 48000), 20)); LFC = np.log(room.FC)
def bins(v): return np.stack([np.interp(LF, LFC, v[i]) for i in range(len(v))])
def one(d):
    c = truth.load(d); t = truth.room_truth(c); wet = c['wet']; direct, rev = truth.parts(c)
    tau = np.where(np.isfinite(t['tau']), t['tau'], np.nanmedian(t['tau'])); kap = np.where(np.isfinite(t['kappa']), t['kappa'], np.nanmedian(t['kappa']))
    if not np.isfinite(tau).all(): return None
    sc0 = score.Scorer(direct); dw = sc0(wet); out = {}; n = len(wet)
    def sc(y): return 100.0 * (1.0 - sc0(y) / max(dw, 1e-9))
    X = dv.fwd(wet); E = bandpow(X); Es = dv.smooth_tf(E); Rt = bandpow(dv.fwd(rev)); St = bandpow(dv.fwd(direct))
    out['exact bins'] = sc(dv.inv(X * np.sqrt(St / np.maximum(St + Rt, 1e-30))[None], n))
    Eb = E @ BM.T; Rb = Rt @ BM.T; Sb = St @ BM.T
    out['true fraction per band'] = sc(dv.inv(X * dv.gain_from(Es, bins(Rb / np.maximum(Sb + Rb, 1e-30)) * Es, 0.0)[None], n))
    A = kap / tau
    for beta in (1.0, 1.5, 2.0, 3.0, 5.0):
        Rm = dv.reverb_power(Es, tau, A * beta); out['bins, room x%.1f inside' % beta] = sc(dv.inv(X * dv.gain_from(Es, Rm, 0.0)[None], n))
    for beta in (1.0, 1.5, 2.0, 3.0, 5.0):
        Rh = np.stack([room.reverb_track(Eb[:, b], tau[b], A[b] * beta, dv.DT) for b in range(len(tau))], 1)
        out['bands, room x%.1f inside' % beta] = sc(dv.inv(X * dv.gain_from(Es, bins(np.minimum(Rh / np.maximum(Eb, 1e-30), 1.0)) * Es, 0.0)[None], n))
    # how the band tracker's reverb fraction compares with the true one, in true tails and where the direct sound leads
    Rh = np.stack([room.reverb_track(Eb[:, b], tau[b], A[b], dv.DT) for b in range(len(tau))], 1); fr = np.minimum(Rh / np.maximum(Eb, 1e-30), 2.0); ft = Rb / np.maximum(Sb + Rb, 1e-30)
    live = Eb > Eb.max(0, keepdims=True) * 1e-5; tl = live & (ft > 0.9); dr = live & (ft < 0.3)
    out['_tail: predicted/true dB'] = float(np.median(10 * np.log10(fr[tl] / ft[tl] + 1e-9))) if tl.sum() > 10 else np.nan
    out['_direct: predicted fraction'] = float(np.median(fr[dr])) if dr.sum() > 10 else np.nan
    out['_direct: true fraction'] = float(np.median(ft[dr])) if dr.sum() > 10 else np.nan
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
        for i, k in enumerate(keys): print('    %-28s median %5.1f  mean %5.1f  worse than untouched in %2.0f%%' % (k, np.nanmedian(a[sel, 2 + i]), np.nanmean(a[sel, 2 + i]), 100 * np.mean(a[sel, 2 + i] < 0)))
    print('%.0f s' % (time.time() - t0))
