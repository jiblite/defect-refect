# Envelopes and the true room numbers of every reverb test case, computed once: the estimators can then be tried in seconds.
import sys, json, glob, os
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, soundfile as sf, room, truth
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/'; OUT = '/tmp/claude-0/v2/proto/cache/'
M = room.band_matrix(2048)
def coh(x):
    """Left-right coherence per band over the whole file (1 = both channels carry the same thing in that band)."""
    L = room.stft(x[:, 0], 2048, 512); R = room.stft(x[:, 1], 2048, 512)
    cross = np.abs((L * np.conj(R)).sum(0)); pl = (np.abs(L) ** 2).sum(0); pr = (np.abs(R) ** 2).sum(0)
    return (M @ cross) / np.sqrt((M @ pl) * (M @ pr) + 1e-30)
def one(d):
    name = os.path.basename(d.rstrip('/')); fam = os.path.basename(os.path.dirname(d.rstrip('/'))); o = OUT + fam + '/' + name + '.npz'
    if os.path.exists(o): return name
    out = {}
    for w in 'AB':
        c = truth.load(d, w); t = truth.room_truth(c)
        for key, x in [(w + 'w', c['wet']), (w + 'd', c['dry'])]:
            e = room.envelopes(x); out[key + '_slow'] = e['slow'].astype(np.float32); out[key + '_fast'] = e['fast'].astype(np.float32); out[key + '_coh'] = coh(x)
        out[w + '_S'] = t['S'].astype(np.float32); out[w + '_R'] = t['R'].astype(np.float32); out[w + '_kappa'] = t['kappa']
        direct, rev = truth.parts(c); out[w + '_cohrev'] = coh(rev)
    out['true_tau'] = t['tau']; out['kappa_click'] = t['kappa_click']; out['fixed_db'] = c['meta']['fixed_db']
    os.makedirs(OUT + fam, exist_ok=True); np.savez_compressed(o, **out); return name
if __name__ == '__main__':
    ds = sorted(glob.glob(S + sys.argv[1] + '/*/'))
    with Pool(4) as p:
        for i, n in enumerate(p.imap_unordered(one, ds)):
            if i % 30 == 0: print(i, n, flush=True)
