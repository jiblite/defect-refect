# Ceilings: what the exact true magnitude response achieves in different representations (grid, FIR length, phase, caps, linked or dual).
import sys, json, time
from multiprocessing import Pool
import numpy as np
from base import *
import fir
VARIANTS = {}
def V(name, **kw): VARIANTS[name] = kw
# grid: ('lin', nfft) or ('log', per_oct); n: taps; phase; cap: (boost, cut) dB; dual: per-channel curves
V('lin2.9_n32k_min_nocap', grid=('lin', 16384), n=32768, phase='minimum', cap=(90, 90), dual=True)
V('lin2.9_n32k_min_cap24', grid=('lin', 16384), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('lin2.9_n32k_min_cap18', grid=('lin', 16384), n=32768, phase='minimum', cap=(18, 18), dual=True)
V('lin2.9_n32k_min_cap12', grid=('lin', 16384), n=32768, phase='minimum', cap=(12, 12), dual=True)
V('lin2.9_n32k_min_cap24_linked', grid=('lin', 16384), n=32768, phase='minimum', cap=(24, 24), dual=False)
V('lin2.9_n32k_lin_cap24', grid=('lin', 16384), n=32768, phase='linear', cap=(24, 24), dual=True)
V('lin2.9_n8k_min_cap24', grid=('lin', 16384), n=8192, phase='minimum', cap=(24, 24), dual=True)
V('lin2.9_n2k_min_cap24', grid=('lin', 16384), n=2048, phase='minimum', cap=(24, 24), dual=True)
V('lin5.9_n32k_min_cap24', grid=('lin', 8192), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('lin11.7_n32k_min_cap24', grid=('lin', 4096), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('lin23.4_n32k_min_cap24', grid=('lin', 2048), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('log48_n32k_min_cap24', grid=('log', 48), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('log24_n32k_min_cap24', grid=('log', 24), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('log12_n32k_min_cap24', grid=('log', 12), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('log6_n32k_min_cap24', grid=('log', 6), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('log3_n32k_min_cap24', grid=('log', 3), n=32768, phase='minimum', cap=(24, 24), dual=True)
V('log24_n8k_min_cap24', grid=('log', 24), n=8192, phase='minimum', cap=(24, 24), dual=True)
V('log24_n8k_min_cap24_linked', grid=('log', 24), n=8192, phase='minimum', cap=(24, 24), dual=False)
V('log24_n8k_lin_cap24', grid=('log', 24), n=8192, phase='linear', cap=(24, 24), dual=True)
def grid_of(g): return fir.lin_grid(g[1]) if g[0] == 'lin' else fir.log_grid(g[1])
def true_curve(c, g, dual=True):
    f, db = true_db(c); grid = grid_of(g); cv = np.stack([cell_mean(f, db[o], grid) for o in range(2)])
    if not dual: cv = np.repeat(cv.mean(0, keepdims=True), 2, 0)
    return grid, cv
def one(d):
    a = load(d, 'A'); b = load(d, 'B'); name = a['name']; sa = score.Scorer(a['dry']); sb = score.Scorer(b['wet']); dwa = sa(a['wet']); dwb = sb(b['dry'])
    row = dict(name=name, effect=a['meta']['effect'], split=a['meta']['split'])
    # the exact complex inverse (a little regularised) as the top of the ladder
    H = true_H(a); n = len(a['wet']); N = int(2 ** np.ceil(np.log2(n + NT))); Hn = np.stack([np.fft.rfft(a['h'][o][:NT, o], N) for o in range(2)])
    out = np.stack([np.fft.irfft(np.fft.rfft(a['wet'][:, o], N) * np.conj(Hn[o]) / (np.abs(Hn[o]) ** 2 + 1e-6), N)[:n] for o in range(2)], 1)
    row['rm:exact_complex'] = 100 * (1 - sa(out) / dwa)
    for vn in VSEL:
        v = VARIANTS[vn]; grid, cv = true_curve(a, v['grid'], v['dual']); boost, cut = v['cap']
        inv = np.clip(-cv, -cut, boost); h = fir.to_ir(grid, inv, v['n'], v['phase']); out = fir.apply_ir(a['wet'], h, v['phase']); row['rm:' + vn] = 100 * (1 - sa(out) / dwa)
        fw = np.clip(cv, -cut, boost); h = fir.to_ir(grid, fw, v['n'], v['phase']); out = fir.apply_ir(b['dry'], h, v['phase']); row['rf:' + vn] = 100 * (1 - sb(out) / dwb)
    return row
if __name__ == '__main__':
    VSEL = sys.argv[2].split(',') if len(sys.argv) > 2 else list(VARIANTS); tag = sys.argv[1]
    t0 = time.time()
    with Pool(2) as pool: rows = pool.map(one, case_dirs(), chunksize=2)
    json.dump(rows, open(WORK + 'ceil_%s.json' % tag, 'w'))
    for k in [k for k in rows[0] if k[:3] in ('rm:', 'rf:')]:
        table([dict(name=r['name'], effect=r['effect'], split=r['split'], v=r[k]) for r in rows], k)
    print('%.0f s' % (time.time() - t0))
else:
    VSEL = list(VARIANTS)
