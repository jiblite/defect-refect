# Decay time the classical way, guided by the mask: straight-line fits to the level (dB) in stretches the mask calls "all room".
import sys, os, glob, json
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, torch, net
torch.set_num_threads(1)
tr = np.load('roomtruth_sb.npz'); names = list(tr['names'])
def slopes(LE, g, gmax=-10.0, minlen=8, skip=2):
    """Per band: list of (slope dB/s, weight) from runs of frames with gain under gmax."""
    T, B = LE.shape; fl = LE.max() - 57.0; out = [[] for _ in range(B)]; tt = np.arange(T) * cm.DT
    for b in range(B):
        m = (g[:, b] < gmax) & (LE[:, b] > fl); t = 0
        while t < T:
            if not m[t]: t += 1; continue
            e = t
            while e < T and m[e]: e += 1
            a = t + skip
            if e - a >= minlen:
                y = LE[a:e, b]; x = tt[a:e]; p = np.polyfit(x, y, 1); res = np.sqrt(np.mean((np.polyval(p, x) - y) ** 2)); rng = -p[0] * (x[-1] - x[0])
                if p[0] < -3.0 and rng > 6.0: out[b].append((p[0], rng / (1.0 + res)))
            t = e
    return out
def t60_from(sl):
    T60 = np.full(len(sl), np.nan)
    for b, lst in enumerate(sl):
        if not lst: continue
        s = np.array([v[0] for v in lst]); w = np.array([v[1] for v in lst]); o = np.argsort(s); cw = np.cumsum(w[o]) / w.sum(); T60[b] = -60.0 / s[o][np.searchsorted(cw, 0.5)]
    return T60
def run(mode, model=None):
    rows = []
    for i, nm in enumerate(names):
        z = np.load('sb/%s.npz' % nm); LE = z['LE'].astype(np.float64); LS = z['LS'].astype(np.float64)
        if mode == 'true': g = np.clip(np.maximum(LS, LE.max() - 60) - LE, -30, 0)
        else:
            with torch.no_grad(): g = model(torch.from_numpy(net.inputs(z['LE'].astype(np.float32), z['U64'].astype(np.float32), z['U256'].astype(np.float32)))[None])[0].numpy().astype(np.float64)
        T60 = t60_from(slopes(LE, g)); ok = tr['ok'][i] & np.isfinite(tr['T60'][i]) & np.isfinite(T60)
        info = json.loads(str(z['info']))
        if ok.sum() >= 3: rows.append((info['split'], np.median(np.log2(T60[ok] / tr['T60'][i][ok])), ok.sum(), info['drr']))
        else: rows.append((info['split'], np.nan, ok.sum(), info['drr']))
    for sp in ('dev', 'test'):
        v = np.array([r[1] for r in rows if r[0] == sp]); f = np.isfinite(v); print('%-6s %-5s | answered %3.0f%% | decay time x%.2f off (median), within 20%%: %3.0f%%, within 35%%: %3.0f%%, bias x%.2f' % (mode, sp, 100 * f.mean(), 2 ** np.median(np.abs(v[f])), 100 * np.mean(np.abs(v[f]) < np.log2(1.2)), 100 * np.mean(np.abs(v[f]) < np.log2(1.35)), 2 ** np.median(v[f])))
run('true')
ck = torch.load('unet_b7.pt', map_location='cpu', weights_only=False); m = net.UNet(ch=tuple(ck['cfg']['ch'])); m.load_state_dict(ck['state']); m.eval(); run('unet', m)
