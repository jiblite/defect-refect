# All 120 scoreboard reverbs: the first look alone against the pair (whole / sound itself / ring-out as heard), and how much deeper the pair goes.
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, predict, render
import depth2_test as D
from multiprocessing import Pool
PAIRS = [('unet_x', 'unet_x+unet_xp2_e5'), ('unet_y1', 'unet_y1+unet_y1p2')]; SR = 48000
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; n = len(wet); L = 6 * SR; ft = cm.features(wet, direct); LE = ft['LE']; nb = int(6.0 / cm.DT)
    Cw = D.cells(direct, wet); base = [D.seg(Cw, 0, n), D.seg(Cw, 0, L), D.seg(Cw, L, n)]; live = LE > LE.max() - 50.0; E = 10 ** (LE / 10)
    r = dict(name=c['name'], base=base, drr=float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20))))
    for M in set(sum(PAIRS, ())):
        g = predict.gains(ft, M); C = D.cells(direct, render.render(wet, g, 0.0)); b = live[:nb]; gb = g[:nb]
        r[M] = dict(sc=[100.0 * (1.0 - D.seg(C, a, e) / max(w, 1e-9)) for (a, e), w in zip(((0, n), (0, L), (L, n)), base)], mean=float(gb[b].mean()), ecut=float(10 * np.log10((E[:nb] * 10 ** (gb / 10)).sum() / E[:nb].sum())), deep=float((gb[b] < -6).mean()))
    return r
if __name__ == '__main__':
    ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))
    with Pool(4) as p: rows = p.map(one, ds)
    json.dump(rows, open('/tmp/claude-0/v2/train/first_vs_pair.json', 'w')); print(len(rows))
    for f, pr in PAIRS:
        for i, part in ((0, 'whole'), (1, 'sound itself'), (2, 'ring-out')):
            a = np.array([r[f]['sc'][i] for r in rows]); b = np.array([r[pr]['sc'][i] for r in rows])
            print('%-18s %-12s first look: median %5.1f mean %6.1f worse %2.0f%% | pair: median %5.1f mean %6.1f worse %2.0f%% | pair better on %d, worse by 10+ on %d' % (pr, part, np.median(a), a.mean(), 100 * (a < 0).mean(), np.median(b), b.mean(), 100 * (b < 0).mean(), (b > a).sum(), (b - a <= -10).sum()))
