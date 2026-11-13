# How simple may the room be? The true energy response (read off the click responses) is replaced by simpler descriptions and
# each is scored the usual way: room built from the description, put on dry sound B, compared with the real B.
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, score, roomcap as rc, rc_look
from multiprocessing import Pool
K = rc.K; k = np.arange(K)
def t60_of(w):
    e = np.cumsum(w[::-1])[::-1]
    if e[0] <= 0: return np.nan
    L = 10 * np.log10(e / e[0] + 1e-30)
    for a0, a1 in ((-5.0, -25.0), (-3.0, -15.0), (-2.0, -10.0)):
        i0 = int(np.argmax(L < a0)); i1 = int(np.argmax(L < a1))
        if i1 > i0 + 2 and L.min() < a1:
            sl = np.polyfit(np.arange(i0, i1) * cm.DT, L[i0:i1], 1)[0]
            if sl < 0: return -60.0 / sl
    return np.nan
def expo(kap, T60, k0=0):
    """Per band: nothing before lag k0, then a plain exponential decay with decay time T60 and total energy kap."""
    W = np.zeros((cm.NB, K))
    for b in range(cm.NB):
        if not np.isfinite(T60[b]) or kap[b] <= 0: continue
        r = 10 ** (-6.0 * cm.DT / T60[b]); w = np.where(k >= k0, r ** (k - k0), 0.0); W[b] = kap[b] * w / w.sum()
    return W
def smooth_bands(v, n=2):
    """Median over +-n bands (in the log domain for strengths)."""
    out = np.array(v, float)
    for b in range(len(v)): seg = v[max(0, b - n):b + n + 1]; seg = seg[np.isfinite(seg)]; out[b] = np.median(seg) if len(seg) else np.nan
    return out
def one(d):
    cA = truth.load(d, 'A'); Wc = rc_look.click_W(cA); kap = Wc.sum(1); T60 = np.array([t60_of(w) for w in Wc]); T60 = np.where(np.isfinite(T60), T60, np.nanmedian(T60) if np.isfinite(T60).any() else 0.3)
    dA, rA = truth.parts(cA); ft = cm.features(cA['wet'], dA); U = 10 ** (np.clip(ft['U256'], -40, 0) / 10); g = np.clip(np.maximum(ft['LS'], ft['LE'].max() - 60) - ft['LE'], -30, 0); tail = (g < -12.0) & (ft['LE'] > ft['LE'].max() - 50.0)
    u = np.array([np.median(U[tail[:, b], b]) if tail[:, b].sum() >= 10 else np.nan for b in range(cm.NB)]); u = np.where(np.isfinite(u), u, np.nanmedian(u) if np.isfinite(u).any() else 1.0); u = np.clip(u * 1.15, 0.0, 1.0)
    cB = truth.load(d, 'B'); sc0 = score.Scorer(cB['wet']); base = sc0(cB['dry']); out = {}
    def sc(W, uu=u): return 100.0 * (1.0 - sc0(rc.apply_room(cB['dry'], rc.synth_ir(W, uu, seed=3), 1.0)) / max(base, 1e-9))
    out['the true response'] = sc(Wc)
    # where the response starts: first lag holding 5 % of the energy, all bands together
    tot = Wc.sum(0); cs = np.cumsum(tot) / max(tot.sum(), 1e-30); k0 = int(np.argmax(cs > 0.05)); pk = int(np.argmax(tot))
    out['strength + decay time per band'] = sc(expo(kap, T60, 0)); out['+ start lag'] = sc(expo(kap, T60, k0)); out['+ start at the peak'] = sc(expo(kap, T60, pk))
    # early part (first 85 ms) kept as it is, tail as an exponential with the tail's own strength
    e8 = Wc.copy(); e8[:, 8:] = 0.0; lk = Wc[:, 8:].sum(1); We = e8 + np.array([np.concatenate([np.zeros(8), expo(lk, T60, 0)[b][:K - 8]]) for b in range(cm.NB)]); out['true first 85 ms + exponential tail'] = sc(We)
    # smoother over frequency: medians over +-2 bands, and a 6-number description (strength and decay time at low / mid / high)
    out['strength + decay, smoothed over 5 bands'] = sc(expo(10 ** smooth_bands(np.log10(kap + 1e-12)), smooth_bands(T60), k0))
    lf = np.log(cm.FC41); an = np.log(np.array([125.0, 1000.0, 8000.0])); grp = [cm.FC41 < 350, (cm.FC41 >= 350) & (cm.FC41 < 2800), cm.FC41 >= 2800]
    k3 = np.array([np.median(np.log10(kap[g_] + 1e-12)) for g_ in grp]); t3 = np.array([np.median(np.log(T60[g_])) for g_ in grp]); out['strength + decay at low / mid / high only'] = sc(expo(10 ** np.interp(lf, an, k3), np.exp(np.interp(lf, an, t3)), k0))
    out['one strength, one decay time'] = sc(expo(np.full(cm.NB, np.median(kap[(cm.FC41 > 350) & (cm.FC41 < 5600)])), np.full(cm.NB, np.median(T60[(cm.FC41 > 350) & (cm.FC41 < 5600)])), k0))
    # early part as one flat step holding the true early energy (first 85 ms), then the exponential tail with the tail's own strength
    ek = Wc[:, :8].sum(1); Wf = np.zeros((cm.NB, K)); Wf[:, :8] = ek[:, None] / 8.0; Wf[:, 8:] = expo(lk, T60, 0)[:, :K - 8]; out['flat early step + exponential tail'] = sc(Wf)
    # the same with the early step relative to what the plain exponential would hold there, one number for all bands
    ex = expo(kap, T60, 0); ratio = np.median((ek / np.maximum(ex[:, :8].sum(1), 1e-30))[(cm.FC41 > 350) & (cm.FC41 < 5600)]); Wg = ex.copy(); Wg[:, :8] *= ratio; Wg *= (kap / np.maximum(Wg.sum(1), 1e-30))[:, None]; out['exponential with one early factor'] = sc(Wg); out['_early factor dB'] = float(10 * np.log10(ratio + 1e-9))
    # how exact must the numbers be? the per-band description with errors
    for lab, dk, ft_ in (('strength +2 dB', 2.0, 1.0), ('strength -2 dB', -2.0, 1.0), ('strength +4 dB', 4.0, 1.0), ('strength -4 dB', -4.0, 1.0), ('decay time x1.25', 0.0, 1.25), ('decay time /1.25', 0.0, 0.8), ('decay time x1.5', 0.0, 1.5), ('decay time /1.5', 0.0, 1 / 1.5)):
        out[lab] = sc(expo(kap * 10 ** (dk / 10.0), T60 * ft_, 0))
    kt = (10 ** (ft['LR'] / 10)).sum(0) / np.maximum((10 ** (ft['LS'] / 10)).sum(0), 1e-30); mid = (cm.FC41 > 450) & (cm.FC41 < 7000)
    return cA['name'], float(-10 * np.log10(np.mean(kt[mid]) + 1e-9)), out
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '2'))) as p: rows = p.map(one, ds)
    drr = np.array([r[1] for r in rows])
    for kk in rows[0][2]:
        if kk.startswith('_'): print(kk, np.round([r[2][kk] for r in rows], 1)); continue
        v = np.array([r[2][kk] for r in rows]); print('%-44s median %5.1f, below zero %2.0f%% | light %5.1f | middle %5.1f | wet %5.1f' % (kk, np.median(v), 100 * np.mean(v < 0), np.median(v[drr >= 8]), np.median(v[(drr < 8) & (drr >= 0)]), np.median(v[drr < 0])))
    print('%.0f s' % (time.time() - t0))
