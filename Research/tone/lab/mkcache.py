# Feature cache: long-term statistics of every dry training clip and of random mixes of two or three of them.
import sys, glob, json, os, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
from multiprocessing import Pool
import feat
WORK = '/tmp/claude-0/v2/tone/lab/work/'
FILES = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))
def rd(p):
    x = sf.read(p, always_2d=True)[0].astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    return x[:, :2]
def one(p): return feat.feats(rd(p))
def mix(seed):
    r = np.random.default_rng(50000 + seed); k = int(r.choice([2, 2, 3])); idx = r.choice(len(FILES), k, replace=False); y = 0
    for i in idx:
        x = rd(FILES[i]); x = np.roll(x, int(r.integers(0, len(x))), 0) * 10 ** (r.uniform(-12, 0) / 20) / (np.sqrt((x ** 2).mean()) + 1e-9)
        if r.random() < 0.3: x = x[:, ::-1]
        y = y + x[:288000] if not np.isscalar(y) else x[:288000]
    return feat.feats(y)
if __name__ == '__main__':
    t0 = time.time(); what = sys.argv[1]
    with Pool(2) as pool:
        if what == 'clips':
            F = pool.map(one, FILES, chunksize=8); np.save(WORK + 'train_feats.npy', np.array(F, np.float32)); json.dump([os.path.basename(p) for p in FILES], open(WORK + 'train_files.json', 'w'))
        else:
            a, b = int(sys.argv[2]), int(sys.argv[3]); F = pool.map(mix, range(a, b), chunksize=8); np.save(WORK + 'mix_feats_%d.npy' % a, np.array(F, np.float32))
    print(what, len(F), time.time() - t0, flush=True)
