import sys, os, glob, json, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
from multiprocessing import Pool
import feat, echo2
def one(a):
    p, seed = a; x = sf.read(p, always_2d=True)[0].astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    P = feat.frames(x[:, :2]); act = feat.active(P); Q = (P[0] + P[1])[act]; r = np.random.default_rng(seed); out = [(echo2.find4(Q, fit=False), echo2.find4(Q))]
    for g in (0.3, 0.45, 0.7):
        d = float(np.exp(r.uniform(np.log(0.6 * 48), np.log(12 * 48)))); sg = g * float(r.choice([-1, 1])); Qc = Q * np.exp(2 * echo2.comb_ln(d, sg))[None, :]; out.append((d, sg, echo2.find4(Qc, fit=False), echo2.find4(Qc)))
    return out
if __name__ == '__main__':
    if sys.argv[1] == 'dev':
        from base import *
        for d in case_dirs():
            c = load(d, 'A'); m = c['meta']
            if m['effect'] != 'tone_comb': continue
            P = feat.frames(c['wet']); act = feat.active(P); Q = (P[0] + P[1])[act]; f0 = echo2.find4(Q, fit=False); f1 = echo2.find4(Q)
            print('%s true d %.2f g %+.3f | projection only: %s | with fit: %s' % (c['name'], m['params']['ms'] * 48, m['params']['g'], ' '.join('d %.2f g %+.3f' % t for t in f0), ' '.join('d %.2f g %+.3f' % t for t in f1)))
    else:
        files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[int(sys.argv[3])::int(sys.argv[2])]; t0 = time.time()
        with Pool(2) as pool: R = pool.map(one, [(p, i) for i, p in enumerate(files)], chunksize=4)
        kind = np.array([os.path.basename(p).split('_')[1] for p in files]); fa = np.array([len(r[0][1]) > 0 for r in R])
        print('clean clips %d: false alarms %.1f%% (%s)' % (len(files), 100 * fa.mean(), ' '.join('%s %.0f%%' % (k[:9], 100 * fa[kind == k].mean()) for k in sorted(set(kind)))))
        print('false alarm gains:', np.round(sorted([abs(r[0][1][0][1]) for r in R if r[0][1]]), 2))
        for j, g in enumerate((0.3, 0.45, 0.7)):
            for nm, col in (('projection', 2), ('fit', 3)):
                near = [(r[j + 1][col][0][0] - r[j + 1][0], r[j + 1][col][0][1] / r[j + 1][1]) for r in R if len(r[j + 1][col]) > 0 and abs(r[j + 1][col][0][0] - r[j + 1][0]) < 2.0]
                de = np.abs([n_[0] for n_ in near]); ge = [n_[1] for n_ in near]; other = np.mean([len(r[j + 1][col]) > 0 and abs(r[j + 1][col][0][0] - r[j + 1][0]) >= 2.0 for r in R])
                print('g=%.2f %-10s: found (within 2 samples) %.0f%%, something else first %.0f%% | delay error samples: median %.3f, 75th %.3f, 90th %.3f, share over 0.25: %.0f%% | gain ratio median %.2f (10th..90th %.2f..%.2f)' % (g, nm, 100 * len(near) / len(R), 100 * other, *np.percentile(de, [50, 75, 90]), 100 * (de > 0.25).mean(), *np.percentile(ge, [50, 10, 90])))
        print('%.0f s' % (time.time() - t0))
