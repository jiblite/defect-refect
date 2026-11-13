# Trains the small neural "room or sound" model on the training cases; checks every epoch on the scoreboard's reverb cases.
import sys, glob, json, time, os
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, torch, net
torch.set_num_threads(int(os.environ.get('THREADS', '3'))); torch.manual_seed(0); np.random.seed(0)
TAG = os.environ.get('TAG', 'unet'); EPOCHS = int(os.environ.get('EPOCHS', '24')); BS = int(os.environ.get('BS', '8')); LR = float(os.environ.get('LR', '2e-3')); CH = tuple(int(v) for v in os.environ.get('CH', '12,16,24,32,48,48').split(','))
PASS2 = os.environ.get('PASS2', '') == '1'; DROPS = os.environ.get('DROPS', '') == '1'
def load(files):
    X, LE, LS, top, w, meta = [], [], [], [], [], []
    for f in files:
        z = np.load(f); le, ls = z['LE'].astype(np.float32), z['LS'].astype(np.float32); x = net.inputs(le, z['U64'].astype(np.float32), z['U256'].astype(np.float32), DROPS)
        if PASS2: x = np.concatenate([x, np.load(f.replace('/cases/', '/pass2_cases/').replace('/sb/', '/pass2_sb/'))['X'].astype(np.float32)])
        X.append(x); LE.append(le); LS.append(ls); top.append(le.max())
        p, base = net.proxy(np.zeros_like(le), le.astype(np.float64), ls.astype(np.float64)); w.append(1.0 / max(base, 1.0)); meta.append(json.loads(str(z['info'])))
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(LE)), torch.from_numpy(np.stack(LS)), torch.tensor(top), torch.tensor(w, dtype=torch.float32), meta
t0 = time.time(); tr = load(sorted(glob.glob('/tmp/claude-0/v2/train/cases/*.npz'))); sb = load(sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz'))); print('loaded', tr[0].shape, sb[0].shape, '%.0f s' % (time.time() - t0), flush=True)
GLOB = int(os.environ.get('GLOB', '0')); model = net.UNet(cin=(7 if PASS2 else 4) + (2 if DROPS else 0), ch=CH, glob=GLOB); print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
INIT = os.environ.get('INIT'); TOT = float(os.environ.get('TOT', '0')); PCT = float(os.environ.get('PCT', '0.1')); BCE = float(os.environ.get('BCE', '3.0'))
if INIT:
    st = torch.load('/tmp/claude-0/v2/train/%s.pt' % INIT, map_location='cpu', weights_only=False)['state']; k0 = 'enc.0.0.weight'
    if st[k0].shape[1] < model.state_dict()[k0].shape[1]:        # a second pass starts as the first pass: the extra pictures enter with weight zero
        w = torch.zeros_like(model.state_dict()[k0]); w[:, :st[k0].shape[1]] = st[k0]; st[k0] = w
    miss = model.load_state_dict(st, strict=False); print('starting from', INIT, '| new parts:', sorted(set(k.split('.')[0] for k in miss.missing_keys)), flush=True); assert not miss.unexpected_keys
opt = torch.optim.Adam(model.parameters(), LR); N = len(tr[0]); steps = EPOCHS * (N // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=PCT, div_factor=20.0)
def evaluate():
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(sb[0]), 16):
            g = model(sb[0][i:i + 16]).numpy()
            for k in range(len(g)):
                p, base = net.proxy(g[k].astype(np.float64), sb[1][i + k].numpy().astype(np.float64), sb[2][i + k].numpy().astype(np.float64)); m = sb[5][i + k]; out.append((m['split'], m['drr'], p))
    model.train(); r = {}
    for sp in ('dev', 'test'):
        v = np.array([o[2] for o in out if o[0] == sp]); d = np.array([o[1] for o in out if o[0] == sp]); r[sp] = (np.median(v), np.mean(v), 100 * np.mean(v < 0), np.median(v[d >= 8]), np.median(v[d < 0]))
    return r, out
best = -1e9; step = 0
for ep in range(EPOCHS):
    perm = torch.randperm(N); tot = 0.0; te = time.time()
    for i in range(0, N - BS + 1, BS):
        idx = perm[i:i + BS]; g, z = model(tr[0][idx]); l = (net.train_loss(g, z, tr[1][idx], tr[2][idx], tr[3][idx], BCE).mean((1, 2)) * tr[4][idx]).mean()
        if TOT > 0: l = l + TOT * net.total_loss(g, tr[1][idx], tr[2][idx], tr[3][idx]).mean()
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step(); tot += float(l.detach()); step += 1
        if step == 20: print('  20 steps in %.0f s' % (time.time() - te), flush=True)
    r, out = evaluate(); sc = r['dev'][0] + r['dev'][1] + r['test'][0] * 0     # model choice uses the dev reverbs only
    print('epoch %2d  loss %.4f  %.0f s | dev: median %.1f mean %.1f below zero %.0f%% (light %.1f, wet %.1f) | unseen: median %.1f mean %.1f below zero %.0f%% (light %.1f, wet %.1f)' % ((ep, tot / (N // BS), time.time() - te) + r['dev'] + r['test']), flush=True)
    torch.save(dict(cfg=model.cfg, state=model.state_dict(), kind='unet'), '/tmp/claude-0/v2/train/%s_last.pt' % TAG)
    if sc > best: best = sc; torch.save(dict(cfg=model.cfg, state=model.state_dict(), kind='unet'), '/tmp/claude-0/v2/train/%s.pt' % TAG); json.dump(out, open('/tmp/claude-0/v2/train/%s.sb.json' % TAG, 'w'))
print('done, %.0f s' % (time.time() - t0))
