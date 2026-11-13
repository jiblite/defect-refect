import sys, os, glob, json, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
from multiprocessing import Pool
import feat, echo2
WORK = '/tmp/claude-0/v2/tone/lab/work/'
def clean_one(a):
    p, seed = a; x = sf.read(p, always_2d=True)[0].astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    P = feat.frames(x[:, :2]); act = feat.active(P); Q = (P[0] + P[1])[act]; r = np.random.default_rng(seed); out = [echo2.find3(Q)]
    for g in (0.3, 0.45, 0.7):
        d = float(np.exp(r.uniform(np.log(0.6 * 48), np.log(12 * 48)))); sg = g * float(r.choice([-1, 1])); out.append((d, sg, echo2.find3(Q * np.exp(2 * echo2.comb_ln(d, sg))[None, :])))
    return out
if __name__ == '__main__':
    if sys.argv[1] == 'dev':
        from base import *
        for d in case_dirs():
            c = load(d, 'A'); m = c['meta']
            if m['effect'] not in ('tone_comb', 'tone_box'): continue
            print(c['name'], m['params'] if m['effect'] == 'tone_comb' else '', flush=True)
            for key in ('wet', 'dry'):
                P = feat.frames(c[key]); act = feat.active(P); t0 = time.time(); f = echo2.find3((P[0] + P[1])[act], verbose=(m['effect'] == 'tone_comb' and key == 'wet' and len(sys.argv) > 2))
                print('  %s: %s  (%.2f s)' % (key, ' '.join('d %.2f smp = %.3f ms g %+.3f;' % (dd, dd / 48, g) for dd, g in f) or 'nothing', time.time() - t0), flush=True)
    else:
        files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[int(sys.argv[3])::int(sys.argv[2])]; t0 = time.time()
        with Pool(2) as pool: R = pool.map(clean_one, [(p, i) for i, p in enumerate(files)], chunksize=4)
        kind = np.array([os.path.basename(p).split('_')[1] for p in files]); fa = np.array([len(r[0]) > 0 for r in R])
        s = 'clean clips %d: false alarms %.1f%% (%s)' % (len(files), 100 * fa.mean(), ' '.join('%s %.0f%%' % (k[:9], 100 * fa[kind == k].mean()) for k in sorted(set(kind)))); print(s)
        print('false alarm gains:', np.round(sorted([abs(r[0][0][1]) for r in R if r[0]]), 2))
        for j, g in enumerate((0.3, 0.45, 0.7)):
            first = np.array([len(r[j + 1][2]) > 0 and abs(r[j + 1][2][0][0] - r[j + 1][0]) < 0.6 for r in R]); anyw = np.array([any(abs(dd - r[j + 1][0]) < 0.6 for dd, gg in r[j + 1][2]) for r in R]); wrong = np.array([len(r[j + 1][2]) > 0 and abs(r[j + 1][2][0][0] - r[j + 1][0]) >= 0.6 for r in R])
            err = [r[j + 1][2][0][1] / r[j + 1][1] for r in R if len(r[j + 1][2]) > 0 and abs(r[j + 1][2][0][0] - r[j + 1][0]) < 0.6]
            print('g=%.2f: found first %.0f%%, found at all %.0f%%, something else first %.0f%%; found gain / true gain: median %.2f, 10th..90th pct %.2f..%.2f' % (g, 100 * first.mean(), 100 * anyw.mean(), 100 * wrong.mean(), *np.percentile(err, [50, 10, 90])))
        print('%.0f s' % (time.time() - t0))
