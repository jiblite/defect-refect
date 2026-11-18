# All 120 scoreboard reverbs: the room numbers each FIRST look's answer gives (what the second look is fed), against the truth.
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, predict, roomnum, rc_look, param_ceiling as pc
from multiprocessing import Pool
MODELS = os.environ.get('MODELS', 'unet_x,unet_y1').split(',')
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; m = c['meta']; ft = cm.features(wet, direct); mid = (cm.FC41 > 450) & (cm.FC41 < 7000)
    kt = 10 * np.log10((10 ** (ft['LR'] / 10)).sum(0) / np.maximum((10 ** (ft['LS'] / 10)).sum(0), 1e-30) + 1e-9); Tc = np.array([pc.t60_of(x) for x in rc_look.click_W(c)]); okb = (10 ** (ft['LS'] / 10)).max(0) > 10 ** ((ft['LE'].max() - 40.0) / 10); mm = mid & okb & np.isfinite(Tc)
    r = dict(name=c['name'], drr=float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20))), A=m['A'], Tt=float(np.median(Tc[mm])) if mm.any() else None, kt=float(np.median(kt[mm])) if mm.any() else None)
    for M in MODELS:
        g = predict.gains(ft, M); rn = roomnum.room_numbers(ft['LE'], g)
        if rn is None: r[M] = None; continue
        kdb, T60, ev = rn; T, ev0 = roomnum.decay_times(ft['LE'], g)
        r[M] = dict(T=float(np.median(T60[mm])) if mm.any() else float(np.median(T60)), k=float(np.median(kdb[mm])) if mm.any() else float(np.median(kdb)), bands=int(np.isfinite(T).sum()), room=float((g < -10).mean()))
    return r
if __name__ == '__main__':
    ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))
    with Pool(4) as p: rows = p.map(one, ds)
    json.dump(rows, open('/tmp/claude-0/v2/train/nums_all.json', 'w')); print(len(rows), 'cases')
    for M in MODELS:
        f = [r for r in rows if r[M] is not None and r['Tt']]; te = np.array([np.log2(r[M]['T'] / r['Tt']) for r in f]); ke = np.array([r[M]['k'] - r['kt'] for r in f])
        print('%-8s found %d of %d | decay time: median off x%.2f, within 20%%: %.0f%%, more than 1.5 times too long: %d, more than 2 times: %d, too short by 1.5: %d | strength: bias %+.1f dB, median |error| %.1f dB, off by more than 6 dB: %d | bands with a free decay (median) %d' % (
            M, len(f), len(rows), 2 ** np.median(np.abs(te)), 100 * np.mean(np.abs(te) < np.log2(1.2)), (te > np.log2(1.5)).sum(), (te > 1).sum(), (te < -np.log2(1.5)).sum(), np.median(ke), np.median(np.abs(ke)), (np.abs(ke) > 6).sum(), np.median([r[M]['bands'] for r in f])))
