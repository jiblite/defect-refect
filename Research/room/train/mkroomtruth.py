# The true room numbers of every training case and scoreboard sound, per band: strength (reverb energy per unit of direct
# energy, as it happened on the sound) and decay time (from the energy response fitted to the true direct and reverb parts).
import sys, os, glob, json, time
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, roomcap as rc
from multiprocessing import Pool
def one(f):
    z = np.load(f); LE = z['LE'].astype(np.float64); LS = z['LS'].astype(np.float64); LR = z['LR'].astype(np.float64); fl = LE.max() - 60.0
    S = 10 ** (LS / 10); R = 10 ** (LR / 10); kap = R.sum(0) / np.maximum(S.sum(0), 1e-30); kdb = 10 * np.log10(kap + 1e-6)
    ok = (S.max(0) > 10 ** ((fl + 20.0) / 10)) & (S.sum(0) > 1e-3 * S.sum(0).max())                 # the band's direct sound comes within 40 dB of the loudest cell
    T60 = np.full(cm.NB, np.nan)
    if np.median(kdb[ok]) > -40.0 if ok.any() else False:
        W = rc.energy_response(S, R, c1=0.0, tie=0, iters=4, fill=False); T60 = rc.numbers(W)['T60']
    return os.path.basename(f)[:-4], kdb.astype(np.float32), T60.astype(np.float32), ok
if __name__ == '__main__':
    t0 = time.time()
    for lab in ('cases', 'sb'):
        fs = sorted(glob.glob('/tmp/claude-0/v2/train/%s/*.npz' % lab))
        with Pool(int(os.environ.get('NP', '2'))) as p: rows = p.map(one, fs, chunksize=8)
        np.savez('/tmp/claude-0/v2/train/roomtruth_%s.npz' % lab, names=np.array([r[0] for r in rows]), kdb=np.stack([r[1] for r in rows]), T60=np.stack([r[2] for r in rows]), ok=np.stack([r[3] for r in rows]))
        k = np.stack([r[1] for r in rows]); t = np.stack([r[2] for r in rows]); o = np.stack([r[3] for r in rows]); print(lab, len(rows), 'cases | bands usable %.0f%% | decay time found in %.0f%% of usable bands | median strength %.1f dB, decay time %.2f s | %.0f s' % (100 * o.mean(), 100 * np.isfinite(t[o]).mean(), np.median(k[o]), np.nanmedian(t[o]), time.time() - t0), flush=True)
