# Features of the scoreboard sounds (wet and dry, A and B) and the true curves on the grid, for quick curve-level checks.
import os; os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, time
from multiprocessing import Pool
from base import *
import feat
def true_grid(c):
    f, db = true_db(c); df = f[1]; e = feat._e; lo = np.clip(np.ceil(e[:-1] / df).astype(int), 0, len(f)); hi = np.clip(np.ceil(e[1:] / df).astype(int), 0, len(f)); out = []
    for o in range(2):
        cs = np.concatenate([[0.0], np.cumsum(db[o])]); v = np.interp(feat.GRID, f, db[o]); w = hi - lo >= 2; v[w] = (cs[hi[w]] - cs[lo[w]]) / (hi[w] - lo[w]); out.append(v)
    return np.array(out)
def one(d):
    a = load(d, 'A'); b = load(d, 'B')
    return dict(name=a['name'], effect=a['meta']['effect'], split=a['meta']['split'], Aw=feat.feats(a['wet']), Ad=feat.feats(a['dry']), Bw=feat.feats(b['wet']), Bd=feat.feats(b['dry']), C=true_grid(a).astype(np.float32))
if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as pool: R = pool.map(one, case_dirs(), chunksize=2)
    np.savez(WORK + 'bench_feats.npz', names=[r['name'] for r in R], effects=[r['effect'] for r in R], splits=[r['split'] for r in R], **{k: np.array([r[k] for r in R]) for k in ('Aw', 'Ad', 'Bw', 'Bd', 'C')})
    print(time.time() - t0)
