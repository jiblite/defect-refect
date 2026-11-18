# How deep does the answer go where the room is on its own (the ring-out, last 3 s of the scoreboard's reverb pictures)?
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, predict
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_x+unet_xp2_e5')
def one(f):
    z = np.load(f); ft = {k: z[k].astype(np.float64) for k in ('LE', 'U64', 'U256')}; LE = ft['LE']; LS = z['LS'].astype(np.float64); g = predict.gains(ft, MODEL); m = json.loads(str(z['info']))
    fl = LE.max() - 60.0; t = slice(570, None); live = (LE[t] > fl + 3.0) & (LS[t] < fl); E = 10 ** (LE[t] / 10.0)
    if live.sum() < 50: return None
    gl = g[t][live]; w = E[live]; lev = (LE[t] - LE.max())[live]
    r = dict(drr=m['drr'], split=m['split'], n=int(live.sum()), med=float(np.median(gl)), p25=float(np.percentile(gl, 25)), p75=float(np.percentile(gl, 75)), ew=float(10 * np.log10((w * 10 ** (gl / 10.0)).sum() / w.sum())), shallow=float((gl > -6.0).mean()), shallow_e=float(w[gl > -6.0].sum() / w.sum()))
    for lo, hi in ((-25, 0), (-40, -25), (-57, -40)):
        s = (lev >= lo) & (lev < hi); r['med %d..%d' % (lo, hi)] = float(np.median(gl[s])) if s.sum() > 20 else None
    # the first 0.3 s of the ring-out against the rest
    first = np.zeros(LE[t].shape, bool); first[:28] = True; a = live & first; b = live & ~first
    r['med first'] = float(np.median(g[t][a])) if a.sum() > 20 else None; r['med later'] = float(np.median(g[t][b])) if b.sum() > 20 else None
    return r
if __name__ == '__main__':
    with Pool(2) as p: R = [r for r in p.map(one, sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz')), chunksize=4) if r]
    A = lambda k, rs=R: np.array([r[k] for r in rs if r[k] is not None])
    print('%d pictures with a ring-out; cells where the room is on its own and 3 dB+ over the floor' % len(R))
    print('gain there: median of medians %.1f dB (quartiles of the per-sound median: %.1f, %.1f); energy-weighted %.1f dB; share of cells shallower than 6 dB: %.0f%% (of the energy: %.0f%%)' % (np.median(A('med')), np.percentile(A('med'), 25), np.percentile(A('med'), 75), np.median(A('ew')), 100 * np.median(A('shallow')), 100 * np.median(A('shallow_e'))))
    for k in ('med -25..0', 'med -40..-25', 'med -57..-40', 'med first', 'med later'): print('  %-14s median gain %.1f dB (n=%d)' % (k, np.median(A(k)), len(A(k))))
    for lab, sel in (('heavy', lambda r: r['drr'] < 5), ('light', lambda r: r['drr'] >= 5)):
        rs = [r for r in R if sel(r)]; print('  %s: median gain %.1f dB, energy-weighted %.1f dB' % (lab, np.median(A('med', rs)), np.median(A('ew', rs))))
