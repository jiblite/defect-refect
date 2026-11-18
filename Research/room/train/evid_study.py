# Where does the reader's evidence for a decay time sit (low bands under 450 Hz, mid bands, high bands over 7 kHz), and how does
# the pair do by that pattern? Scoreboard pictures in five forms and pictures without a reverb.
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
T = '/tmp/claude-0/v2/train/'; FIRST = os.environ.get('FIRST', 'unet_y1'); PAIR = os.environ.get('PAIR', 'unet_y1+unet_y1p2')
def score(g, LE, LS):
    fl = LE.max() - 60.0; ref = np.maximum(LS, fl); a = np.minimum(np.abs(np.maximum(LE + g, fl) - ref), 30.0).sum(); b = np.minimum(np.abs(np.maximum(LE, fl) - ref), 30.0).sum(); return float(100.0 * (1.0 - a / b)) if b > 1e-9 else None
def taken(g, LE): E = 10 ** (LE / 10); return float(10 * np.log10(max(1.0 - (E * 10 ** (g / 10)).sum() / E.sum(), 1e-6)))
def one(f):
    import predict, roomnum, common as cm
    z = np.load(f); ft = {k: z[k].astype(np.float64) for k in ('LE', 'U64', 'U256')}; LE = ft['LE']; LS = z['LS'].astype(np.float64); m = json.loads(str(z['info'])); dry = float(m.get('drr', 99.0)) > 90
    g1 = predict.gains(ft, FIRST); g2 = predict.gains(ft, PAIR); live = LE > LE.max() - 50.0
    r = dict(file=os.path.basename(f), set=os.path.basename(os.path.dirname(f)), split=m.get('split', 'train'), drr=float(m.get('drr', 99.0)), kind=m.get('kind', ''), dry=dry, excess=float((g1 - g2)[live].mean()))
    val = (lambda g: taken(g, LE)) if dry else (lambda g: score(g, LE, LS)); r['first'] = val(g1); r['pair'] = val(g2)
    lo = cm.FC41 < 450; hi = cm.FC41 > 7000; mid = ~lo & ~hi; r['ev'] = {}
    for gmax, minlen in ((-10.0, 8), (-7.0, 6), (-4.0, 5)):
        sl = roomnum.decay_slopes(LE, g1, gmax, minlen); has = np.array([bool(l) for l in sl]); r['ev']['%g' % gmax] = [int(has[lo].sum()), int(has[mid].sum()), int(has[hi].sum())]
    rn = roomnum.room_numbers(LE, g1); r['found'] = rn is not None
    if rn is not None: r['T'] = [float(np.median(rn[1][s])) for s in (lo, mid, hi)]; r['k'] = [float(np.median(rn[0][s])) for s in (lo, mid, hi)]
    return r
if __name__ == '__main__':
    t0 = time.time(); fs = []
    for c in ('sb', 'sb_n', 'sb_m', 'sb_c', 'sb_e', 'sb_cl', 'sb_fx'): fs += sorted(glob.glob(T + c + '/*.npz'))
    with Pool(4) as p: rows = p.map(one, fs, chunksize=4)
    json.dump(rows, open(T + 'evid_study_%s.json' % FIRST, 'w')); print(len(rows), '%.0f s' % (time.time() - t0))
