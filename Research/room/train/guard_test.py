# A guard on the second look: where it goes deeper than the first look by more than D dB on average (over the cells within 50 dB
# of the loudest), its answer is pulled back towards the first look's so that the average excess is D. Scored as heard on
# the 120 scoreboard reverbs (whole file / the sound itself / the ring-out), for both pairs.
#   NP=4 python3 guard_test.py           -> guard_test.json
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, predict, render
import depth2_test as D2
from multiprocessing import Pool
PAIRS = [('unet_x', 'unet_x+unet_xp2_e5'), ('unet_y1', 'unet_y1+unet_y1p2')]; SR = 48000; DS = (None, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0)
def excess(g1, g2, LE): live = LE > LE.max() - 50.0; return float((g1 - g2)[live].mean())
def guarded(g1, g2, LE, D):
    if D is None: return g2
    e = excess(g1, g2, LE); return g2 if e <= D else g1 + (D / e) * (g2 - g1)
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; n = len(wet); L = 6 * SR; ft = cm.features(wet, direct); LE = ft['LE']
    Cw = D2.cells(direct, wet); base = [D2.seg(Cw, 0, n), D2.seg(Cw, 0, L), D2.seg(Cw, L, n)]
    r = dict(name=c['name'], base=base, drr=float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20))))
    for f, pr in PAIRS:
        g1 = predict.gains(ft, f); g2 = predict.gains(ft, pr); r[pr] = dict(excess=excess(g1, g2, LE), sc={})
        for D in DS:
            if D is not None and r[pr]['excess'] <= D: r[pr]['sc'][str(D)] = r[pr]['sc']['None']; continue
            C = D2.cells(direct, render.render(wet, guarded(g1, g2, LE, D), 0.0)); r[pr]['sc'][str(D)] = [100.0 * (1.0 - D2.seg(C, a, e) / max(w, 1e-9)) for (a, e), w in zip(((0, n), (0, L), (L, n)), base)]
    return r
if __name__ == '__main__':
    ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))
    with Pool(int(os.environ.get('NP', '4'))) as p: rows = p.map(one, ds)
    json.dump(rows, open('/tmp/claude-0/v2/train/guard_test.json', 'w'))
    for f, pr in PAIRS:
        ex = np.array([r[pr]['excess'] for r in rows]); print('%s | the second look deeper than the first by (dB, average): median %.2f, 90th %.2f, 95th %.2f, largest %s' % (pr, np.median(ex), np.percentile(ex, 90), np.percentile(ex, 95), ' '.join('%.1f' % v for v in np.sort(ex)[::-1][:6])))
        for D in DS:
            s = '  guard at %-4s (on %2d cases)' % (D, 0 if D is None else (ex > D).sum())
            for i, part in ((0, 'whole'), (1, 'sound itself'), (2, 'ring-out')):
                v = np.array([r[pr]['sc'][str(D)][i] for r in rows]); b = np.array([r['base'][i] for r in rows]); dd = -b * v / 100
                s += ' | %s: median %5.1f mean %6.1f worse %2.0f%%, further by 0.5 dB+: %d, by 1 dB+: %d' % (part, np.median(v), v.mean(), 100 * (v < 0).mean(), (dd >= 0.5).sum(), (dd >= 1).sum())
            print(s)
