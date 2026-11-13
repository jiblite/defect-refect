# The reader's own bias, measured on the training cases with the true mask: found strength and decay time against the truth.
import sys, os, glob, json, time
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, roomnum
from multiprocessing import Pool
tr = np.load('roomtruth_cases.npz'); names = list(tr['names'])
def one(i):
    z = np.load('cases/%s.npz' % names[i]); LE = z['LE'].astype(np.float64); LS = z['LS'].astype(np.float64); g = np.clip(np.maximum(LS, LE.max() - 60) - LE, -30, 0)
    T60, ev = roomnum.decay_times(LE, g); info = json.loads(str(z['info']))
    if not np.isfinite(T60).any(): return (i, info['engine'], None, None, None)
    Ts = roomnum.smooth_fill(T60, ev); kap, ek = roomnum.strengths(LE, g, Ts); return (i, info['engine'], T60, Ts, 10 * np.log10(kap))
if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as p: rows = p.map(one, range(len(names)), chunksize=16)
    mid = (cm.FC41 > 450) & (cm.FC41 < 7000); K, Tm, Tt, Kt, eng = [], [], [], [], []
    for i, e, T60, Ts, kdb in rows:
        if e == 'none' or T60 is None: continue
        ok = tr['ok'][i] & np.isfinite(tr['T60'][i]) & np.isfinite(kdb) & (tr['kdb'][i] > -40)
        if ok.sum() < 3: continue
        K.append(np.median((kdb - tr['kdb'][i])[ok])); Tm.append(np.median(np.log2(Ts[ok] / tr['T60'][i][ok]))); Tt.append(np.median(tr['T60'][i][ok])); Kt.append(np.median(tr['kdb'][i][ok])); eng.append(e)
    K, Tm, Tt, Kt, eng = map(np.array, (K, Tm, Tt, Kt, eng)); dry = [r for r in rows if r[1] == 'none']
    print('wet cases answered: %d of %d | dry cases answered: %d of %d' % (len(K), sum(1 for r in rows if r[1] != 'none'), sum(1 for r in dry if r[2] is not None), len(dry)))
    print('strength bias %+.2f dB (spread: quartiles %+.1f .. %+.1f) | decay time bias x%.3f, off x%.2f, within 20%% %.0f%%' % (np.median(K), np.percentile(K, 25), np.percentile(K, 75), 2 ** np.median(Tm), 2 ** np.median(np.abs(Tm)), 100 * np.mean(np.abs(Tm) < np.log2(1.2))))
    for lab, sel in (('T60 < 0.5', Tt < 0.5), ('0.5-1', (Tt >= 0.5) & (Tt < 1)), ('1-2', (Tt >= 1) & (Tt < 2)), ('2-4', (Tt >= 2) & (Tt < 4)), ('> 4', Tt >= 4), ('strength < -8 dB', Kt < -8), ('-8..0', (Kt >= -8) & (Kt < 0)), ('> 0', Kt >= 0)):
        if sel.sum(): print('  %-18s n=%4d | strength bias %+.2f dB | decay time bias x%.2f, within 20%% %.0f%%' % (lab, sel.sum(), np.median(K[sel]), 2 ** np.median(Tm[sel]), 100 * np.mean(np.abs(Tm[sel]) < np.log2(1.2))))
    for e in sorted(set(eng)):
        sel = eng == e; print('  %-20s n=%4d | strength bias %+.2f dB | decay time bias x%.2f, within 20%% %.0f%%' % (e, sel.sum(), np.median(K[sel]), 2 ** np.median(Tm[sel]), 100 * np.mean(np.abs(Tm[sel]) < np.log2(1.2))))
    print('%.0f s' % (time.time() - t0))
