# Per test case, at the de-reverb's own frame rate: band power of the wet sound, of its true direct part and true reverb part,
# and the "not shared between the channels" power under a few ways of averaging. For trying mask estimators in seconds.
import sys, json, glob, os
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, scipy.signal as ss, room, truth, dereverb as dv
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/'; OUT = '/tmp/claude-0/v2/proto/cache2/'; BM = room.band_matrix(dv.NF)
corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
def bp(X): return (np.abs(X[0]) ** 2 + np.abs(X[1]) ** 2) @ BM.T
def unshared(X, tavg, favg=2, bil=0.0):
    """Band power the channels do not share, (frames, bands). bil > 0: frames are weighted by how close their level is to the
    centre frame's (scale bil dB), so a loud note does not leak into the estimate for the quiet moments next to it."""
    pl = np.abs(X[0]) ** 2; pr = np.abs(X[1]) ** 2; c = X[0] * np.conj(X[1]); T, F = pl.shape; k = max(1, int(round(tavg / dv.DT))); h = k // 2
    kf = np.ones((1, 2 * favg + 1)) / (2 * favg + 1); fs = lambda a: ss.fftconvolve(a, kf, 'same')
    if bil <= 0:
        kt = np.ones((k, 1)) / k; ts = lambda a: ss.fftconvolve(a, kt, 'same'); pls, prs, cs = fs(ts(pl)), fs(ts(pr)), fs(ts(c.real)) + 1j * fs(ts(c.imag))
    else:
        Lv = 10 * np.log10(fs(pl + pr) + 1e-20); pls = np.zeros((T, F)); prs = np.zeros((T, F)); cs = np.zeros((T, F), complex); ws = np.zeros((T, F))
        for o in range(-h, h + 1):
            lo, hi = max(0, -o), min(T, T - o); w = np.exp(-np.abs(Lv[lo + o:hi + o] - Lv[lo:hi]) / bil)
            pls[lo:hi] += w * pl[lo + o:hi + o]; prs[lo:hi] += w * pr[lo + o:hi + o]; cs[lo:hi] += w * c[lo + o:hi + o]; ws[lo:hi] += w
        pls, prs, cs = fs(pls / ws), fs(prs / ws), fs((cs / ws).real) + 1j * fs((cs / ws).imag)
    lam2 = np.maximum((pls + prs) - np.sqrt(np.maximum((pls - prs) ** 2 + 4 * np.abs(cs) ** 2, 0.0)), 0.0)
    return lam2 @ BM.T, (pls + prs) @ BM.T
VAR = {'p064': (0.064, 2, 0.0), 'p128': (0.128, 2, 0.0), 'p256': (0.256, 2, 0.0), 'b128': (0.128, 2, 6.0), 'b256': (0.256, 2, 6.0), 'b256w': (0.256, 4, 6.0)}
def one(d):
    name = os.path.basename(d.rstrip('/')); o = OUT + name + '.npz'
    if os.path.exists(o): return name
    c = truth.load(d); t = truth.room_truth(c); direct, rev = truth.parts(c); X = dv.fwd(c['wet']); out = dict(E=bp(X), S=bp(dv.fwd(direct)), R=bp(dv.fwd(rev)), tau=t['tau'], kappa=t['kappa'])
    for k, (ta, fa, bl) in VAR.items():
        D, P = unshared(X, ta, fa, bl); out['D_' + k] = D.astype(np.float32); out['P_' + k] = P.astype(np.float32)
    Dr, Pr = unshared(dv.fwd(rev), 0.128); Dd, Pd = unshared(dv.fwd(direct), 0.128)
    out['u_rev'] = Dr.sum(0) / np.maximum(Pr.sum(0), 1e-30); out['u_src'] = Dd.sum(0) / np.maximum(Pd.sum(0), 1e-30); out['kind'] = corp[c['meta']['A']]['kind']
    for k in ('E', 'S', 'R'): out[k] = out[k].astype(np.float32)
    os.makedirs(OUT, exist_ok=True); np.savez_compressed(o, **out); return name
if __name__ == '__main__':
    ds = sorted(glob.glob(S + 'reverb/*/'))
    with Pool(4) as p:
        for i, n in enumerate(p.imap_unordered(one, ds)):
            if i % 30 == 0: print(i, n, flush=True)
