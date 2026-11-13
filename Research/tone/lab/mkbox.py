# A bank of 'small box' responses (the dev-split zoo effect tone_box with its own sampler), as dB curves per channel on the dense FFT grid.
import sys, os, time
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np
from multiprocessing import Pool
SR = 48000
def one(seed):
    import zoo
    rng = np.random.default_rng(1000 + seed); p = zoo.ZOO['tone_box'][2](rng)
    h = zoo._room_ir(dict(p, rt60=p['rt60']))[: int(0.03 * SR)].copy(); h[int(0.02 * SR):] *= np.linspace(1, 0, len(h) - int(0.02 * SR))[:, None]
    return h.astype(np.float32)
if __name__ == '__main__':
    n = int(sys.argv[1]); t0 = time.time()
    with Pool(2) as pool: H = pool.map(one, range(n), chunksize=4)
    L = max(len(h) for h in H); out = np.zeros((n, L, 2), np.float32)
    for i, h in enumerate(H): out[i, :len(h)] = h
    np.save('/tmp/claude-0/v2/tone/lab/work/box_bank.npy', out); print(out.shape, time.time() - t0)
