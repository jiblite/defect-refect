# For choosing the guard on the second look (see guard_bands.py): per picture the score along the whole line from the first
# look's answer (w = 0) to the pair's (w = 1), the excess (how much deeper the pair goes on average) and how many mid bands
# (450 Hz .. 7 kHz) gave the reader a free decay of their own. Any rule w(excess, mid bands) can then be read off the file.
#   FIRST=unet_y1 PAIR=unet_y1+unet_y1p2 NP=4 python3 guard_bands2.py [per training set, default 500]   -> guard_bands2_<FIRST>.json
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
T = '/tmp/claude-0/v2/train/'; FIRST = os.environ.get('FIRST', 'unet_y1'); PAIR = os.environ.get('PAIR', 'unet_y1+unet_y1p2'); WS = [round(0.1 * i, 1) for i in range(11)]
def score(g, LE, LS):
    fl = LE.max() - 60.0; ref = np.maximum(LS, fl); a = np.minimum(np.abs(np.maximum(LE + g, fl) - ref), 30.0).sum(); b = np.minimum(np.abs(np.maximum(LE, fl) - ref), 30.0).sum(); return float(100.0 * (1.0 - a / b)) if b > 1e-9 else None
def taken(g, LE): E = 10 ** (LE / 10); return float(10 * np.log10(max(1.0 - (E * 10 ** (g / 10)).sum() / E.sum(), 1e-6)))
def one(f):
    import predict, roomnum
    z = np.load(f); ft = {k: z[k].astype(np.float64) for k in ('LE', 'U64', 'U256')}; LE = ft['LE']; LS = z['LS'].astype(np.float64); m = json.loads(str(z['info'])); dry = float(m.get('drr', 99.0)) > 90
    g1 = predict.gains(ft, FIRST); g2 = predict.gains(ft, PAIR); live = LE > LE.max() - 50.0; T60, ev = roomnum.decay_times(LE, g1); rn = roomnum.room_numbers(LE, g1)
    r = dict(file=os.path.basename(f), set=os.path.basename(os.path.dirname(f)), split=m.get('split', 'train'), drr=float(m.get('drr', 99.0)), kind=m.get('kind', ''), dry=dry, excess=float((g1 - g2)[live].mean()),
             found=rn is not None, mid=int(np.isfinite(T60[roomnum.MID]).sum()), bands=int(np.isfinite(T60).sum()), evmid=float(ev[roomnum.MID].sum()), evall=float(ev.sum()))
    val = (lambda g: taken(g, LE)) if dry else (lambda g: score(g, LE, LS)); r['line'] = [val(g1 + w * (g2 - g1)) for w in WS]; return r
if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500; t0 = time.time(); fs = []; rng = np.random.default_rng(5)
    for c in ('sb', 'sb_n', 'sb_m', 'sb_c', 'sb_e', 'sb_cl', 'sb_fx'): fs += sorted(glob.glob(T + c + '/*.npz'))
    for c in ('cases', 'cases_w', 'cases_s', 'cases_x', 'cases_xc', 'cases_xm'):
        a = sorted(glob.glob(T + c + '/*.npz')); fs += [a[i] for i in sorted(rng.permutation(len(a))[:n])]
    with Pool(int(os.environ.get('NP', '4'))) as p: rows = [r for r in p.map(one, fs, chunksize=4) if r]
    json.dump(rows, open(T + 'guard_bands2_%s.json' % FIRST, 'w')); print(len(rows), 'pictures, %.0f s' % (time.time() - t0))
