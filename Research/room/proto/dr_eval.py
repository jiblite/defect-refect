# De-reverb scored end to end on the test cases: how much of the reverb is gone (100 = equals the true dry sound).
import sys, json, glob, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, room, truth, dereverb as dv, score
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/reverb/'; corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
def one(d):
    c = truth.load(d); t = truth.room_truth(c); wet, dry = c['wet'], c['dry']; direct, rev = truth.parts(c)
    tau = np.where(np.isfinite(t['tau']), t['tau'], np.nanmedian(t['tau'])); kap = np.where(np.isfinite(t['kappa']), t['kappa'], np.nanmedian(t['kappa']))
    if not np.isfinite(tau).all(): return None
    dw = score.distance(direct, wet); out = {}
    def sc(y): return 100.0 * (1.0 - score.distance(direct, y) / max(dw, 1e-9))
    # 1: the ceiling: the true reverb power per bin as the mask
    Xr = dv.fwd(rev); Xd = dv.fwd(direct); Rt = np.abs(Xr[0]) ** 2 + np.abs(Xr[1]) ** 2; St = np.abs(Xd[0]) ** 2 + np.abs(Xd[1]) ** 2
    X = dv.fwd(wet); G = np.sqrt(St / np.maximum(St + Rt, 1e-30)); out['ceiling'] = sc(dv.inv(X * G[None], len(wet)))
    # 2: the model with the true room numbers
    out['true numbers'] = sc(dv.process(wet, tau, kap / tau, 0.0))
    # 3: the model with the measured room numbers
    r = room.analyse(wet)
    out['measured'] = sc(dv.process(wet, r['tau'], r['A'], 0.0)) if r['found'] else 0.0
    return c['name'], corp[c['meta']['A']]['kind'], float(np.nanmean(t['drr_db'][(room.FC > 450) & (room.FC < 7000)])), dw, out
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob(S + '*/'))
    if len(sys.argv) > 1 and sys.argv[1].isdigit(): ds = ds[::int(sys.argv[1])]
    with Pool(4) as p: rows = [r for r in p.map(one, ds) if r]
    keys = list(rows[0][4].keys())
    if '-v' in sys.argv:
        for r in rows: print('%-24s %-16s DRR %6.1f  wet is %4.1f dB from dry | ' % r[:4] + ' | '.join('%s %5.1f' % (k, r[4][k]) for k in keys))
    a = np.array([[r[2], r[3]] + [r[4][k] for k in keys] for r in rows])
    for lab, sel in [('all', np.ones(len(a), bool)), ('light (DRR >= 5)', a[:, 0] >= 5), ('middle (-2..5)', (a[:, 0] < 5) & (a[:, 0] >= -2)), ('wet (< -2)', a[:, 0] < -2)]:
        print('%-18s n=%3d  ' % (lab, sel.sum()) + ' | '.join('%s: median %5.1f, mean %5.1f, worse than untouched in %2.0f%%' % (k, np.median(a[sel, 2 + i]), np.mean(a[sel, 2 + i]), 100 * np.mean(a[sel, 2 + i] < 0)) for i, k in enumerate(keys)))
    print('%.0f s' % (time.time() - t0))
