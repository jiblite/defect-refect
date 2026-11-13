# The classic blind estimator, studied scale by scale: split the long-term spectrum into octave-wise detail bands (differences of
# Gaussian smoothings along log frequency); per band, how much of what is seen belongs to the effect's curve and how much to the sound
# itself? Shrinkage (threshold, then gain) per band is fitted on training clips x random curves.
import sys, json, time
import numpy as np
import feat, curves, model
WORK = '/tmp/claude-0/v2/tone/lab/work/'
SIG = [0, 1, 2, 4, 8, 16, 32]              # Gaussian sigmas in grid cells (24 cells = 1 octave)
def gs(v, s):
    if s == 0: return v
    k = int(4 * s); t = np.arange(-k, k + 1); w = np.exp(-0.5 * (t / s) ** 2); w /= w.sum()
    vp = np.concatenate([np.repeat(v[..., :1], k, -1), v, np.repeat(v[..., -1:], k, -1)], -1); return np.apply_along_axis(lambda a: np.convolve(a, w, 'valid'), -1, vp)
def bands(v):
    """(..., G) -> (len(SIG), ..., G): detail bands from fine to coarse; the last one is the broad rest minus its mean."""
    sm = [gs(v, s) for s in SIG]; out = [sm[j] - sm[j + 1] for j in range(len(SIG) - 1)]; out.append(sm[-1] - sm[-1].mean(-1, keepdims=True)); return np.stack(out)
def soft(b, th): return np.sign(b) * np.maximum(np.abs(b) - th, 0.0)
def sample(F, pool, n, seed, null=0.25):
    r = np.random.default_rng(seed); i = r.integers(0, len(F), n); v = r.integers(0, 3, n); X = F[i, v].copy(); k = r.integers(0, len(pool), n)
    P = pool[k] * r.uniform(0.4, 1.1, (n, 1, 1)).astype(np.float32); P[r.random(n) < null] = 0.0
    X[:, 0] += P[:, 1]; X[:, 1] += P[:, 2] - P[:, 1]; X[:, 2] += P[:, 3] - P[:, 1]; return X, P[:, 0], k
def carriers(X):
    return dict(lmean=X[:, 0], floor=X[:, 0] + X[:, 1], mix=X[:, 0] + 0.5 * X[:, 1])
def fit(X, C, W, name):
    """Per band: threshold and gain that minimise the weighted mean abs error against the curve's own band."""
    S = carriers(X)[name]; bw = bands(S); bc = bands(C); par = []; rep = []
    for j in range(len(bw)):
        best = None
        for th in (0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12):
            s = soft(bw[j], th); a = float((s * bc[j] * W).sum() / max((s * s * W).sum(), 1e-9)); a = min(max(a, 0.0), 1.0)
            for aa in (a, 0.75 * a, 0.5 * a):
                e = float((np.abs(aa * s - bc[j]) * W).sum() / W.sum())
                if best is None or e < best[0]: best = (e, th, aa)
        e0 = float((np.abs(bc[j]) * W).sum() / W.sum()); lin = float((bw[j] * bc[j] * W).sum() / (bw[j] ** 2 * W).sum())
        par.append((best[1], best[2])); rep.append((np.sqrt((bw[j] ** 2 * W).sum() / W.sum()), np.sqrt((bc[j] ** 2 * W).sum() / W.sum()), lin, best[1], best[2], best[0], e0))
    return par, rep
def estimate(X, par, name):
    bw = bands(carriers(X)[name]); return sum(a * soft(bw[j], th) for j, (th, a) in enumerate(par))
def l1(pred, C, w):
    e = pred - C; e = e - (e * w).sum(-1, keepdims=True) / np.maximum(w.sum(-1, keepdims=True), 1e-6); return (np.abs(e) * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)
if __name__ == '__main__':
    F = np.load(WORK + 'all_feats.npy')[:1057].astype(np.float32); perm = np.random.default_rng(123).permutation(len(F)); va = perm[:100]; tr = perm[100:]
    pool, ty, names = curves.pool(20000, 1); X, C, k = sample(F[tr], pool, 6000, 0); W = model.weights(X)
    Xv, Cv, kv = sample(F[va], pool, 3000, 1); Wv = model.weights(Xv); nul = np.abs(Cv).max(-1) == 0
    z = np.load(WORK + 'bench_feats.npz'); eff = z['effects']; sp = z['splits']; allpar = {}
    for name in ('lmean', 'floor', 'mix'):
        par, rep = fit(X, C, W, name); allpar[name] = par
        print('carrier %s: per detail band [sigma cells %s]' % (name, ' '.join('%d-%d' % (SIG[j], SIG[j + 1]) for j in range(len(SIG) - 1)) + ' rest'))
        print('  rms of the wet band  ', ' '.join('%5.2f' % r[0] for r in rep)); print('  rms of the curve band', ' '.join('%5.2f' % r[1] for r in rep))
        print('  plain gain           ', ' '.join('%5.2f' % r[2] for r in rep)); print('  threshold dB         ', ' '.join('%5.1f' % r[3] for r in rep)); print('  gain after threshold ', ' '.join('%5.2f' % r[4] for r in rep))
        print('  err after / nothing  ', ' '.join('%4.2f/%4.2f' % (r[5], r[6]) for r in rep))
        p = estimate(Xv, par, name); ev = l1(p, Cv, Wv); e0 = l1(0 * p, Cv, Wv)
        print('  held-out clips: err %.2f dB (nothing %.2f), on clean %.2f | by type: %s' % (ev[~nul].mean(), e0[~nul].mean(), ev[nul].mean(), ' '.join('%s %.2f/%.2f' % (names[t], ev[(ty[kv] == t) & ~nul].mean(), e0[(ty[kv] == t) & ~nul].mean()) for t in range(len(names)))))
        s = '  dev wet real:'
        for e in sorted(set(eff)):
            idx = np.array([i for i in range(len(eff)) if eff[i] == e and sp[i] == 'dev'])
            if not len(idx): continue
            Fw = z['Aw'][idx, 2]; Ct = z['C'][idx].mean(1); w = model.weights(Fw); pw = estimate(Fw, par, name); pd = estimate(z['Ad'][idx, 2], par, name)
            s += ' %s %.2f/%.2f c%.2f' % (e.replace('tone_', '')[:6], l1(pw, Ct, w).mean(), l1(0 * pw, Ct, w).mean(), l1(pd, 0 * pd, model.weights(z['Ad'][idx, 2])).mean())
        print(s, flush=True)
    json.dump(allpar, open(WORK + 'classic_par.json', 'w'))
