# Where does the room capture go wrong? Room strength (reverb energy per unit of direct energy, per band) found three ways,
# against what really happened on the sound.
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, gbfeat, roomcap as rc
from multiprocessing import Pool
VAR = {'old (picked by what was seen)': dict(mode='inst', tie=0), 'expected, c1=1': dict(mode='pred', c1=1.0), 'expected, c1=2': dict(mode='pred', c1=2.0), 'expected, c1=0.5': dict(mode='pred', c1=0.5), 'expected, c1=1, tie 0': dict(mode='pred', c1=1.0, tie=0), 'expected, c1=1, 6 rounds': dict(mode='pred', c1=1.0, iters=6)}
def one(d):
    c = truth.load(d, 'A'); dA, rA = truth.parts(c); ft = cm.features(c['wet'], dA); St = 10 ** (ft['LS'] / 10); Rt = 10 ** (ft['LR'] / 10); mid = (cm.FC41 > 450) & (cm.FC41 < 7000)
    kt = Rt.sum(0) / np.maximum(St.sum(0), 1e-30); out = {}
    def db(k): return float(np.median(10 * np.log10((k[mid] + 1e-9) / (kt[mid] + 1e-9))))
    W0 = rc.energy_response(St, Rt, mode='pred', c1=0.0, tie=0, iters=4); out['true parts, all moments'] = db(W0.sum(1))
    g = gbfeat.target(ft['LE'], ft['LS'])[0]; S, R, va, bo = rc.tracks(ft['LE'], g)
    for k, kw in VAR.items(): out[k] = db(rc.energy_response(S, R, valid=va, band_ok=bo, **kw).sum(1))
    return c['name'], float(-10 * np.log10(np.mean(kt[mid]) + 1e-9)), out
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '2'))) as p: rows = p.map(one, ds)
    drr = np.array([r[1] for r in rows])
    for k in rows[0][2]:
        v = np.array([r[2][k] for r in rows]); print('%-32s strength error: median %+5.1f dB, median |error| %4.1f dB | light %+5.1f | middle %+5.1f | wet %+5.1f' % (k, np.median(v), np.median(np.abs(v)), np.median(v[drr >= 8]), np.median(v[(drr < 8) & (drr >= 0)]), np.median(v[drr < 0])))
    if '-v' in sys.argv:
        for r in rows: print('%-24s DRR %5.1f | ' % r[:2] + ' | '.join('%+5.1f' % r[2][k] for k in r[2]))
    print('%.0f s' % (time.time() - t0))
