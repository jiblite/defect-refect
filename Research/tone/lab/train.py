# Trains the tone model on dry-clip statistics plus random curves (the curve is added to the lmean row; the other rows do not move).
import sys, os, time, json
os.environ.setdefault('OMP_WAIT_POLICY', 'passive'); os.environ.setdefault('GOMP_SPINCOUNT', '0')      # shared machine: idle threads sleep instead of spinning
import numpy as np, torch
torch.set_num_threads(min(2, int(os.environ.get('TT', '2'))))                                              # never more than 2 threads
import feat, curves, model
WORK = '/tmp/claude-0/v2/tone/lab/work/'
def cfg(**kw):
    c = dict(tag='m1', steps=20000, bs=32, lr=2e-3, ch=40, null=0.25, second=0.25, shift=8, seed=0, huber=1.0, feats='all_feats.npy', npool=40000, loss='l1', drop=0.0, wd=0.0, mixes=1, every=2500, nout=2); c.update(kw); return c
def make_batch(rng, F, pool, c, idx=None, kidx=None):
    """Dry statistics (a random clip or mix, one of its three channel views, shifted a little along the grid) plus a random curve."""
    n = c['bs'] if idx is None else len(idx); i = rng.integers(0, len(F), n) if idx is None else idx; v = rng.integers(0, 3, n)
    X = F[i, v].astype(np.float32)                                    # (n, K, G)
    if c['shift']:
        for j in range(n):
            s = int(rng.integers(-c['shift'], c['shift'] + 1))
            if s > 0: X[j, :, s:] = X[j, :, :-s].copy(); X[j, :, :s] = X[j, :, s:s + 1]
            elif s < 0: X[j, :, :s] = X[j, :, -s:].copy(); X[j, :, s:] = X[j, :, s - 1:s]
    k = rng.integers(0, len(pool), n) if kidx is None else kidx; P = pool[k] * rng.uniform(0.4, 1.1, (n, 1, 1)).astype(np.float32)
    if kidx is None:
        two = rng.random(n) < c['second']; k2 = rng.integers(0, len(pool), n); P[two] += pool[k2[two]] * rng.uniform(0.3, 1.0, (int(two.sum()), 1, 1)).astype(np.float32)
    P[rng.random(n) < c['null']] = 0.0
    X[:, 0] += P[:, 1]; X[:, 1] += P[:, 2] - P[:, 1]; return X, P[:, 0].copy()
def loss_fn(pred, C, w, c):
    """Level-free error of the curve; with the doubt output: Laplace likelihood |e| / b + log b (the net says how sure it is, cell by cell)."""
    lb = None
    if isinstance(pred, tuple): pred, lb = pred
    e = pred - C; e = e - (e * w).sum(-1, keepdim=True) / w.sum(-1, keepdim=True).clamp(min=1e-6)
    if lb is not None: l = e.abs() * torch.exp(-lb) + lb
    elif c['loss'] == 'l1': l = torch.nn.functional.huber_loss(e, torch.zeros_like(e), reduction='none', delta=c['huber'])
    else: l = 0.5 * e ** 2 / 4.0
    return ((l * w).sum(-1) / w.sum(-1).clamp(min=1e-6)).mean()
def l1(pred, C, w):
    e = pred - C; e = e - (e * w).sum(-1, keepdims=True) / np.maximum(w.sum(-1, keepdims=True), 1e-6); return (np.abs(e) * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)
def wmedian_rows(p, w):
    o = np.argsort(p, -1); ps = np.take_along_axis(p, o, -1); cw = np.cumsum(np.take_along_axis(w, o, -1), -1); k = (cw < 0.5 * cw[:, -1:]).sum(-1); return ps[np.arange(len(p)), np.minimum(k, p.shape[1] - 1)]
def shrunk(p, b, w, kap):
    """The curve around its weighted median, shrunk where it is small against the model's own doubt: c * c^2 / (c^2 + (kap * b)^2)."""
    c = p - wmedian_rows(p, w)[:, None]; return c * c * c / (c * c + (kap * b) ** 2 + 1e-12)
def bench_check(m, split='dev'):
    """Curve error (weighted mean abs dB, level-free) on the scoreboard's real wet sounds, per effect, next to predicting nothing; and the
    size of the curve the model finds on the dry sounds (should be near zero)."""
    z = np.load(WORK + 'bench_feats.npz'); eff = z['effects']; sp = z['splits']; out = {}
    for e in sorted(set(eff)):
        idx = np.array([i for i in range(len(eff)) if eff[i] == e and sp[i] == split])
        if not len(idx): continue
        Fw = z['Aw'][idx, 2]; C = z['C'][idx].mean(1); w = model.weights(Fw); p = model.predict(m, Fw); pd = model.predict(m, z['Ad'][idx, 2]); wd = model.weights(z['Ad'][idx, 2])
        out[e] = (float(l1(p, C, w).mean()), float(l1(0 * p, C, w).mean()), float(l1(pd, 0 * pd, wd).mean()))
    return out
def run(c):
    rng = np.random.default_rng(c['seed']); torch.manual_seed(c['seed'])
    F = np.load(WORK + c['feats']); perm = np.random.default_rng(123).permutation(1057); va = perm[:100]; tr = np.concatenate([perm[100:], np.arange(1057, len(F))])       # the mixes (index 1057 and up) may contain held-out clips; the held-out check is on the unmixed clips only
    if not c['mixes']: tr = perm[100:]
    Ftr = F[tr]; Fva = F[va]
    pool, ty, names = curves.pool(c['npool'], c['seed'] + 1); vpool, vty, _ = curves.pool(2000, 999)
    vr = np.random.default_rng(5); vk = vr.integers(0, len(vpool), 3000); VX, VC = make_batch(vr, Fva, vpool, dict(c, shift=0, null=0.25), idx=vr.integers(0, len(Fva), 3000), kidx=vk); VW = model.weights(VX); nul = np.abs(VC).max(-1) == 0
    m = model.Net(ch=c['ch'], drop=c['drop'], nout=c['nout']); print('weights', model.nparams(m), 'train items', len(Ftr), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=c['lr'], weight_decay=c['wd']); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c['lr'], total_steps=c['steps'], pct_start=0.05)
    t0 = time.time(); acc = 0.0; every = c['every']
    for step in range(1, c['steps'] + 1):
        X, C = make_batch(rng, Ftr, pool, c); w = model.weights(X)
        pred = m(torch.from_numpy(model.net_input(X))); loss = loss_fn(pred, torch.from_numpy(C), torch.from_numpy(w.astype(np.float32)), c)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step(); acc += loss.item()
        if step % every == 0 or step == c['steps']:
            m.eval(); p = model.predict(m, VX); ev = l1(p, VC, VW); e0 = l1(0 * p, VC, VW); b = bench_check(m); extra = ''
            if c['nout'] == 2:
                p, bb = model.predict(m, VX, doubt=True)
                for kap in (1.0, 2.0):
                    ps = shrunk(p, bb, VW, kap); es = l1(ps, VC, VW); extra += ' | shrunk x%g: err %.2f clean %.2f' % (kap, es[~nul].mean(), es[nul].mean())
            m.train()
            print('step %5d  loss %.3f  %4.0f s | held-out clips: err %.2f dB (nothing: %.2f) on clean %.2f | %s | dev wet real: %s' % (step, acc / every, time.time() - t0, ev[~nul].mean(), e0[~nul].mean(), ev[nul].mean(),
                  ' '.join('%s %.2f/%.2f' % (names[t], ev[(vty[vk] == t) & ~nul].mean(), e0[(vty[vk] == t) & ~nul].mean()) for t in range(len(names))),
                  ' '.join('%s %.2f/%.2f c%.2f' % (k.replace('tone_', '')[:6], v[0], v[1], v[2]) for k, v in b.items())) + extra, flush=True); acc = 0.0
            torch.save(dict(state=m.state_dict(), cfg=c), WORK + 'net_%s.pt' % c['tag'])
    return m
if __name__ == '__main__':
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split('='); kw[k] = type(cfg()[k])(v) if not isinstance(cfg()[k], str) else v
    run(cfg(**kw))
