# Where is the ceiling? De-reverb with different kinds of perfect knowledge, to see which missing piece costs the most.
import sys, json, glob, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, numba, scipy.signal as ss, room, truth, dereverb as dv, score
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/reverb/'; corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
@numba.njit(cache=True)
def track_fir(E, W):
    """Reverb power per bin from the sound's own past through a measured energy response W (taps, bins): R(t) = sum_k W[k] S(t-1-k)."""
    T, F = E.shape; K = W.shape[0]; R = np.zeros((T, F)); Sx = np.zeros((T, F))
    for t in range(T):
        for f in range(F):
            r = 0.0
            for k in range(min(K, t)): r += W[k, f] * Sx[t - 1 - k, f]
            R[t, f] = r; s = E[t, f] - r; Sx[t, f] = s if s > 0.0 else 0.0
    return R
def bandpow(X): return (np.abs(X[0]) ** 2 + np.abs(X[1]) ** 2)
BM = None
def one(d):
    global BM
    c = truth.load(d); t = truth.room_truth(c); wet, dry = c['wet'], c['dry']; direct, rev = truth.parts(c)
    tau = np.where(np.isfinite(t['tau']), t['tau'], np.nanmedian(t['tau'])); kap = np.where(np.isfinite(t['kappa']), t['kappa'], np.nanmedian(t['kappa']))
    if not np.isfinite(tau).all(): return None
    dw = score.distance(direct, wet); out = {}
    def sc(y): return 100.0 * (1.0 - score.distance(direct, y) / max(dw, 1e-9))
    X = dv.fwd(wet); E = bandpow(X); Es = dv.smooth_tf(E); Rt = bandpow(dv.fwd(rev)); St = bandpow(dv.fwd(direct)); n = len(wet)
    def run(R, Eref=None, fl=-30.0): return sc(dv.inv(X * dv.gain_from(Es if Eref is None else Eref, R, 0.0, fl)[None], n))
    out['exact bins'] = sc(dv.inv(X * np.sqrt(St / np.maximum(St + Rt, 1e-30))[None], n))
    out['exact bins, floor -30'] = sc(dv.inv(X * np.maximum(np.sqrt(St / np.maximum(St + Rt, 1e-30)), 10 ** -1.5)[None], n))
    Rs = dv.smooth_tf(Rt); out['true R smoothed 3x3'] = run(Rs * Es / np.maximum(dv.smooth_tf(St) + Rs, 1e-30))
    Rs = dv.smooth_tf(Rt, 9, 5); out['true R smoothed 9x5'] = run(Rs * Es / np.maximum(dv.smooth_tf(St, 9, 5) + Rs, 1e-30))
    if BM is None: BM = room.band_matrix(dv.NF)
    Rb = Rt @ BM.T; Eb = (St + Rt) @ BM.T; frac = np.stack([np.interp(np.log(np.maximum(np.fft.rfftfreq(dv.NF, 1 / 48000), 20)), np.log(room.FC), (Rb / np.maximum(Eb, 1e-30))[i]) for i in range(len(Rb))])
    out['true R per band only'] = run(frac * Es)
    out['model, true numbers'] = run(dv.reverb_power(Es, tau, kap / tau))
    # the room's true energy response per band (from the click responses), used in place of the one-slope model
    a, b, pk = truth.direct_window(c['h']); hs = c['h'][0] + c['h'][1]; tail = hs.copy(); tail[:b] = 0; dr = np.zeros_like(hs); dr[a:b] = hs[a:b]
    Et = bandpow(dv.fwd(tail[:48000 * 5])) @ BM.T; Ed = bandpow(dv.fwd(dr[:48000 * 5])) @ BM.T; k0 = int(np.argmax(Ed.sum(1)))
    Wb = Et[k0 + 1:k0 + 1 + 400] / np.maximum(Ed.sum(0), 1e-30)[None]; early = Et[:k0 + 1].sum(0) / np.maximum(Ed.sum(0), 1e-30)       # what arrives within the direct sound's own frames
    Wf = np.stack([np.exp(dv.to_bins(np.log(np.maximum(Wb[k], 1e-12)))) for k in range(len(Wb))])
    out['true energy response'] = run(track_fir(np.ascontiguousarray(Es), np.ascontiguousarray(Wf)))
    # stereo: what the channels do not share, as the reverb power (per bin)
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
