# The room-numbers reader against the truth on every scoreboard sound (A and B): strength and decay time per band.
# Truth: strength as it happened on the sound; decay time from the click responses (energy decay curve, -5..-25 dB).
import sys, os, glob, json, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, roomnum, rc_look, param_ceiling as pc
from multiprocessing import Pool
MODE = os.environ.get('GAINS', 'true')
def one(a):
    d, w = a; c = truth.load(d, w); dA, rA = truth.parts(c); ft = cm.features(c['wet'], dA)
    if MODE == 'true': g = np.clip(np.maximum(ft['LS'], ft['LE'].max() - 60) - ft['LE'], -30, 0)
    else: import predict; g = predict.gains(ft)
    Wc = rc_look.click_W(c); Tc = np.array([pc.t60_of(x) for x in Wc]); kt = 10 * np.log10((10 ** (ft['LR'] / 10)).sum(0) / np.maximum((10 ** (ft['LS'] / 10)).sum(0), 1e-30) + 1e-9)
    T60, ev = roomnum.decay_times(ft['LE'], g); rn = roomnum.room_numbers(ft['LE'], g); mid = (cm.FC41 > 450) & (cm.FC41 < 7000); S = 10 ** (ft['LS'] / 10); okb = S.max(0) > 10 ** ((ft['LE'].max() - 40.0) / 10)
    r = dict(name=c['name'] + '_' + w, split=c['meta']['split'], drr=float(-np.mean(kt[mid & okb])) if (mid & okb).any() else np.nan, T60true=float(np.nanmedian(Tc[mid])))
    if rn is None: r.update(ans=False); return r
    kdb, Ts, evn = rn; m = mid & okb & np.isfinite(Tc)
    r.update(ans=True, kerr=float(np.median((kdb - kt)[m])) if m.any() else np.nan, terr=float(np.median(np.log2(Ts[m] / Tc[m]))) if m.any() else np.nan, traw=float(np.nanmedian(np.log2(T60[m] / Tc[m]))) if (m & np.isfinite(T60)).any() else np.nan, nb=int(np.isfinite(T60).sum()),
             kerr_lo=float(np.median((kdb - kt)[(cm.FC41 < 450) & okb])) if ((cm.FC41 < 450) & okb).any() else np.nan, kerr_hi=float(np.median((kdb - kt)[(cm.FC41 > 7000) & okb])) if ((cm.FC41 > 7000) & okb).any() else np.nan, T60found=float(np.median(Ts[mid])))
    return r
if __name__ == '__main__':
    t0 = time.time(); step = int(sys.argv[1]) if len(sys.argv) > 1 else 1; jobs = [(d, w) for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::step] for w in 'AB']
    with Pool(int(os.environ.get('NP', '2'))) as p: rows = p.map(one, jobs)
    json.dump(rows, open('/tmp/claude-0/v2/train/num_diag_%s.json' % MODE, 'w'))
    for lab, sel in (('all', lambda r: True), ('dev', lambda r: r['split'] == 'dev'), ('unseen', lambda r: r['split'] == 'test'), ('light', lambda r: r['drr'] >= 8), ('middle', lambda r: 0 <= r['drr'] < 8), ('wet', lambda r: r['drr'] < 0), ('short (T60 < 0.6 s)', lambda r: r['T60true'] < 0.6), ('long (T60 > 2.5 s)', lambda r: r['T60true'] > 2.5)):
        rs = [r for r in rows if sel(r)]; an = [r for r in rs if r['ans'] and np.isfinite(r.get('kerr', np.nan)) and np.isfinite(r.get('terr', np.nan))]
        if not rs or not an: continue
        ke = np.array([r['kerr'] for r in an]); te = np.array([r['terr'] for r in an])
        print('%-20s n=%3d answered %3.0f%% | strength: bias %+.1f dB, |err| %.1f dB, within 2 dB %2.0f%% | decay time: bias x%.2f, off x%.2f, within 20%% %2.0f%%, within 35%% %2.0f%%' % (lab, len(rs), 100 * len(an) / len(rs), np.median(ke), np.median(np.abs(ke - np.median(ke))), 100 * np.mean(np.abs(ke - np.median(ke)) < 2), 2 ** np.median(te), 2 ** np.median(np.abs(te)), 100 * np.mean(np.abs(te) < np.log2(1.2)), 100 * np.mean(np.abs(te) < np.log2(1.35))))
    if '-v' in sys.argv:
        for r in rows: print('%-26s %-5s DRR %5.1f T60 %.2f | ' % (r['name'], r['split'], r['drr'], r['T60true']) + ('found T60 %.2f (x%.2f, raw x%.2f, %2d bands) strength err %+.1f (low %+.1f, high %+.1f)' % (r['T60found'], 2 ** r['terr'], 2 ** r['traw'], r['nb'], r['kerr'], r['kerr_lo'], r['kerr_hi']) if r['ans'] else 'no answer'))
    print('%.0f s' % (time.time() - t0))
