# Echo finder v2 on the dev comb and box cases (wet and dry), then false alarms / hit rates on clean training clips.
import sys, os, glob, json, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
from multiprocessing import Pool
import feat, echo2
WORK = '/tmp/claude-0/v2/tone/lab/work/'
def clean_one(a):
    p, seed = a; x = sf.read(p, always_2d=True)[0].astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    P = feat.frames(x[:, :2]); act = feat.active(P); Q = (P[0] + P[1])[act]; r = np.random.default_rng(seed); col = []; f0 = echo2.find(Q, thr=lambda d: 0.08 * np.ones_like(np.atleast_1d(d), float), agree_min=0.0, frac2=-9.0, max_echoes=1, collect=col); out = [col]
    for g in (0.3, 0.45, 0.7):
        d = float(np.exp(r.uniform(np.log(0.6 * 48), np.log(12 * 48)))); sg = g * float(r.choice([-1, 1])); col = []; f = echo2.find(Q * np.exp(2 * echo2.comb_ln(d, sg))[None, :], thr=lambda d: 0.08 * np.ones_like(np.atleast_1d(d), float), agree_min=0.0, frac2=-9.0, max_echoes=1, collect=col); out.append((d, sg, col))
    return out
if __name__ == '__main__':
    if sys.argv[1] == 'dev':
        from base import *
        for d in case_dirs():
            c = load(d, 'A'); m = c['meta']
            if m['effect'] not in ('tone_comb', 'tone_box'): continue
            print(c['name'], m['params'] if m['effect'] == 'tone_comb' else '', flush=True)
            for key in ('wet', 'dry'):
                P = feat.frames(c[key]); act = feat.active(P); t0 = time.time(); f = echo2.find((P[0] + P[1])[act], verbose=(m['effect'] == 'tone_comb' and key == 'wet'))
                print('  %s: %s  (%.2f s)' % (key, ' '.join('d %.2f smp = %.3f ms g %+.3f;' % (dd, dd / 48, g) for dd, g in f) or 'nothing', time.time() - t0), flush=True)
    else:
        files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[::int(sys.argv[2])]; t0 = time.time()
        with Pool(2) as pool: R = pool.map(clean_one, [(p, i) for i, p in enumerate(files)], chunksize=4)
        kind = np.array([os.path.basename(p).split('_')[1] for p in files]); json.dump(dict(files=[os.path.basename(p) for p in files], R=R), open(WORK + 'echo3_clean.json', 'w'))
        print(len(files), 'clips', time.time() - t0, 's')
