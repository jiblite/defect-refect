# Trains the moving-tone model on the fit data (mkdata.py); checks every epoch on examples it has not seen.
#   TAG=mv_a EPOCHS=16 python3 train.py /dev/shm/mv/tr[,/dev/shm/mv/tr2] /dev/shm/mv/va
# BAL=1: every example counts by the size of its own curve (a phaser's few dB weigh as much as a sweep's thirty).
import sys, os, glob, json, time, numpy as np, torch
sys.path.insert(0, '/tmp/claude-0/v2/moving'); import pic, net
torch.set_num_threads(int(os.environ.get('THREADS', '4'))); torch.manual_seed(0); np.random.seed(0)
TAG = os.environ.get('TAG', 'mv'); EPOCHS = int(os.environ.get('EPOCHS', '16')); BS = int(os.environ.get('BS', '8')); LR = float(os.environ.get('LR', '2e-3')); CH = tuple(int(v) for v in os.environ.get('CH', '12,16,24,32,48,48').split(','))
WC = float(os.environ.get('WC', '1.0')); BAL = os.environ.get('BAL', '0') == '1'; OUT = '/tmp/claude-0/v2/moving/'
def load(d, limit=None):
    X, G, L, C, I = [], [], [], [], []
    for f in sorted(sum((glob.glob(q + '/*.npz') for q in d.split(',')), [])):
        z = np.load(f); LE = z['LE'].astype(np.float32); X.append(np.stack([pic.inputs(le) for le in LE]).astype(np.float16)); G.append(z['G']); L.append(np.unpackbits(z['live'], axis=-1)[..., :pic.NB]); C.append(z['cls']); I += [json.loads(s) for s in z['info']]
        if limit and sum(len(x) for x in X) >= limit: break
    return torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(G)), torch.from_numpy(np.concatenate(L)), torch.from_numpy(np.concatenate(C)), I
t0 = time.time(); tr = load(sys.argv[1], int(os.environ.get('LIMIT', '0')) or None); va = load(sys.argv[2]); print('loaded', tuple(tr[0].shape), tuple(va[0].shape), '%.0f s' % (time.time() - t0), flush=True)
model = net.UNet(cin=4, ch=CH); print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
INIT = os.environ.get('INIT')
if INIT: model.load_state_dict(torch.load(OUT + INIT + '.pt', map_location='cpu', weights_only=False)['state']); print('starting from', INIT, flush=True)
N = len(tr[0]); steps = EPOCHS * (N // BS); opt = torch.optim.Adam(model.parameters(), LR); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1, div_factor=20.0)
def batch(d, idx): return d[0][idx].float(), d[1][idx].float() / 2.0, d[2][idx].float(), d[3][idx]
def evaluate(d):
    """On unseen examples: the curve's mean error (dB) over the cells the dry sound fills, for sounds with a moving filter and
    without; how often a frame's class is right; how well 'the share of frames with a filter' tells the two kinds of sound apart."""
    model.eval(); eg, ez, acc, sc, has = [], [], [], [], []; kind = {1: [], 2: []}
    with torch.no_grad():
        for i in range(0, len(d[0]), 16):
            x, G, live, cls = batch(d, slice(i, i + 16)); g, c = model(x); g = g * net.GS; p = torch.softmax(torch.cat([torch.zeros_like(c[..., :1]), c], -1), -1)
            for k in range(len(x)):
                w = live[k]; e = float(((g[k] - G[k]).abs() * w).sum() / w.sum().clamp(min=1)); pos = bool(cls[k].any()); (eg if pos else ez).append(e)
                if pos: kind[int(cls[k].max())].append((e, float((G[k].abs() * w).sum() / w.sum().clamp(min=1))))
                acc.append(float((p[k].argmax(-1) == cls[k]).float().mean())); sc.append(float(1.0 - p[k][:, 0].mean())); has.append(pos)
    model.train(); sc = np.array(sc); has = np.array(has); a, b = sc[has], sc[~has]; auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
    ks = np.array(kind[1]).mean(0); kp = np.array(kind[2]).mean(0)
    return dict(sweep=float(ks[0]), sweep0=float(ks[1]), phaser=float(kp[0]), phaser0=float(kp[1]), err_with=float(np.mean(eg)), err_without=float(np.mean(ez)), frames_right=float(np.mean(acc)), auc=auc, found=float((a > 0.5).mean()), false=float((b > 0.5).mean()))
best = 1e9; step = 0
for ep in range(EPOCHS):
    perm = torch.randperm(N); tot = np.zeros(2); te = time.time()
    for i in range(0, N - BS + 1, BS):
        x, G, live, cls = batch(tr, perm[i:i + BS]); g, c = model(x); lg, lc, l = net.loss_fn(g, c, G, live, cls, WC)
        if BAL: w = 1.0 / (0.25 + ((G / net.GS) ** 2).mean((1, 2))); l = (w * lg / 2.25 + WC * lc).mean()
        else: l = l.mean()
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step(); tot += [float(lg.mean().detach()), float(lc.mean().detach())]; step += 1
        if step == 20: print('  20 steps in %.0f s' % (time.time() - te), flush=True)
    r = evaluate(va); k = N // BS
    print('epoch %2d  curve loss %.4f  class loss %.4f  %.0f s | unseen: curve off by %.2f dB with a filter (sweeps %.2f of %.2f, phasers %.2f of %.2f), %.2f without; frames right %.1f%%; AUC %.3f, found %.0f%%, false %.0f%%' % (ep, tot[0] / k, tot[1] / k, time.time() - te, r['err_with'], r['sweep'], r['sweep0'], r['phaser'], r['phaser0'], r['err_without'], 100 * r['frames_right'], r['auc'], 100 * r['found'], 100 * r['false']), flush=True)
    torch.save(dict(cfg=model.cfg, state=model.state_dict()), OUT + TAG + '_last.pt'); sc = r['sweep'] / r['sweep0'] + r['phaser'] / r['phaser0'] + r['err_without'] / 2.0 + 2.0 * (1.0 - r['frames_right'])
    if sc < best: best = sc; torch.save(dict(cfg=model.cfg, state=model.state_dict()), OUT + TAG + '.pt')
print('done, %.0f s' % (time.time() - t0))
