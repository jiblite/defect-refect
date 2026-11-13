# Tone Match on the scoreboard pairs (one process). For every case:
#   remove : curve = match(target A_dry, sound A_wet), put on A_wet, compared with A_dry        (both versions of the same sound are known)
#   refect : curve = match(target A_wet, sound A_dry), put on B_dry, compared with B_wet        (the pair's curve carried to another sound)
#   cross  : curve = match(target A_wet, sound B_dry), put on B_dry: how close does B's long-term spectrum come to A's (dB rms of the
#            difference, both smoothed with the same Detail, where both have sound)?
import sys, os, json, time
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/tmp/claude-0/v2/tone')
import numpy as np
from base import *
import tone
DET = (0.25, 0.5, 0.6, 0.75, 0.9, 1.0)
def resid(a, y, detail):
    Sa = tone.long_term_db(a); Sy = tone.long_term_db(y); k = np.ones(9) / 9; va = np.convolve(Sa, k, 'same'); vy = np.convolve(Sy, k, 'same'); ok = (va > va.max() - 50) & (vy > vy.max() - 50) & (tone.FB >= 40)
    d = tone.smooth(tone.FB, Sa - Sy, detail); d = d - np.median(d[ok]); return float(np.sqrt((d[ok] ** 2).mean()))
def one(d):
    a = load(d, 'A'); b = load(d, 'B'); sa = score.Scorer(a['dry']); sb = score.Scorer(b['wet']); dwa = sa(a['wet']); dwb = sb(b['dry'])
    row = dict(name=a['name'], effect=a['meta']['effect'], split=a['meta']['split'], dwa=dwa, dwb=dwb, rm={}, rf={}, cross={})
    for st in ('pmean', 'lmean'):
        for det in DET:
            if st == 'lmean' and det not in (0.6, 1.0): continue
            g, c = tone.match(a['dry'], a['wet'], det, stat=st); row['rm']['%s %g' % (st, det)] = 100 * (1 - sa(tone.put_curve(a['wet'], g, c, 1.0)) / dwa)
            g, c = tone.match(a['wet'], a['dry'], det, stat=st); row['rf']['%s %g' % (st, det)] = 100 * (1 - sb(tone.put_curve(b['dry'], g, c, 1.0)) / dwb)
    for det in (0.25, 0.6, 1.0):
        g, c = tone.match(a['wet'], b['dry'], det); y = tone.put_curve(b['dry'], g, c, 1.0)
        row['cross']['%g' % det] = (resid(a['wet'], b['dry'], det), resid(a['wet'], y, det), float(np.abs(c).max()), float(10 * np.log10((y ** 2).sum() / (b['dry'] ** 2).sum())))
    return row
if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else 'dev'; every = int(sys.argv[2]) if len(sys.argv) > 2 else 1; t0 = time.time()
    ds = [d for d in case_dirs() if json.load(open(d + 'meta.json'))['split'] == split][::every]; rows = []
    for d in ds: rows.append(one(d)); print(len(rows), '%.0f s' % (time.time() - t0), flush=True)
    json.dump(rows, open(WORK + 'match_%s.json' % split, 'w'))
    effs = sorted(set(r['effect'] for r in rows))
    for key, lab in (('rm', 'Removed % (A wet -> A dry, both known)'), ('rf', 'Refect % (curve of the A pair onto B)')):
        print(lab + ': median per effect | all: median, mean, share below zero'); print('%-12s' % '', ' '.join('%9s' % e.replace('tone_', '')[:9] for e in effs))
        for k in rows[0][key]:
            v = np.array([r[key][k] for r in rows]); print('%-12s' % k, ' '.join('%9.1f' % np.median([r[key][k] for r in rows if r['effect'] == e]) for e in effs), '| %6.1f %6.1f %3.0f%%' % (np.median(v), v.mean(), 100 * (v < 0).mean()))
    print('unrelated pair (A wet as target, B dry as the sound): long-term spectrum difference in dB rms before -> after, largest gain of the curve, level change of the sound')
    for k in rows[0]['cross']:
        v = np.array([r['cross'][k] for r in rows]); print('  detail %-5s before %.2f -> after %.2f dB rms (medians; worst after %.2f) | largest gain median %.1f dB max %.1f | level change median %+.2f dB, range %+.2f .. %+.2f' % (k, np.median(v[:, 0]), np.median(v[:, 1]), v[:, 1].max(), np.median(v[:, 2]), v[:, 2].max(), np.median(v[:, 3]), v[:, 3].min(), v[:, 3].max()))
