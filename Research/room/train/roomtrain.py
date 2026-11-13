# Trains the "what room is this" model on the training cases; checks every epoch on the scoreboard's reverb sounds.
import sys, glob, json, time, os
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, torch, net, roomnet
torch.set_num_threads(int(os.environ.get('THREADS', '2'))); torch.manual_seed(0); np.random.seed(0)
TAG = os.environ.get('TAG', 'room_a'); EPOCHS = int(os.environ.get('EPOCHS', '30')); BS = int(os.environ.get('BS', '16')); LR = float(os.environ.get('LR', '2e-3')); CH = tuple(int(v) for v in os.environ.get('CH', '16,24,32,48,64,64').split(',')); CROP = int(os.environ.get('CROP', '640'))
def load(lab):
    tr = np.load('/tmp/claude-0/v2/train/roomtruth_%s.npz' % lab); X, LS, LR, meta = [], [], [], []
    for nm in tr['names']:
        z = np.load('/tmp/claude-0/v2/train/%s/%s.npz' % (lab, nm)); X.append(np.stack([z['LE'], z['U64'], z['U256']]).astype(np.float32)); LS.append(z['LS'].astype(np.float32)); LR.append(z['LR'].astype(np.float32)); meta.append(json.loads(str(z['info'])))
    return np.stack(X), np.stack(LS), np.stack(LR), tr['kdb'], tr['T60'], tr['ok'], meta
def prep(X):
    """(N, 3, T, B) raw features -> model input (N, 4, T, B); the level plane is relative to the loudest cell of the stretch given."""
    return torch.from_numpy(np.stack([net.inputs(x[0], x[1], x[2]) for x in X]))
t0 = time.time(); trn = load('cases'); sb = load('sb'); print('loaded', trn[0].shape, sb[0].shape, '%.0f s' % (time.time() - t0), flush=True)
model = roomnet.RoomNet(ch=CH); print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
N = len(trn[0]); T = trn[0].shape[2]; steps = EPOCHS * (N // BS); opt = torch.optim.Adam(model.parameters(), LR, weight_decay=0.0); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1, div_factor=20.0)
sbX = prep(sb[0]); sk, st, stok = roomnet.targets(sb[3], sb[4]); sok = sb[5]
def evaluate():
    model.eval(); K, Tt = [], []
    with torch.no_grad():
        for i in range(0, len(sbX), 16): k, t = model(sbX[i:i + 16]); K.append(k.numpy()); Tt.append(t.numpy())
    model.train(); K = np.concatenate(K); Tt = np.concatenate(Tt); r = {}
    for sp in ('dev', 'test'):
        sel = np.array([m['split'] == sp for m in sb[6]]); ek = (K[sel] - sk[sel]) * 10.0; et = Tt[sel] - st[sel]; mk = sok[sel]; mt = sok[sel] & stok[sel]
        # per sound: the median error over its usable bands
        pk = np.array([np.median(ek[i][mk[i]]) for i in range(sel.sum()) if mk[i].any()]); pt = np.array([np.median(et[i][mt[i]]) for i in range(sel.sum()) if mt[i].sum() >= 3])
        r[sp] = (np.median(np.abs(ek[mk])), np.median(pk), np.median(np.abs(pk)), 2 ** np.median(np.abs(et[mt])), 100 * np.mean(np.abs(pt) < np.log2(1.2)), 2 ** np.median(pt))
    return r, K, Tt
best = 1e9
for ep in range(EPOCHS):
    perm = np.random.permutation(N); tot = 0.0; te = time.time()
    for i in range(0, N - BS + 1, BS):
        idx = np.sort(perm[i:i + BS]); o = np.random.randint(0, T - CROP + 1) if CROP < T else 0; sl = slice(o, o + CROP)
        X = prep(trn[0][idx][:, :, sl]); S = 10 ** (trn[1][idx][:, sl].astype(np.float64) / 10); R = 10 ** (trn[2][idx][:, sl].astype(np.float64) / 10)
        kdb = 10 * np.log10(R.sum(1) / np.maximum(S.sum(1), 1e-30) + 1e-6); full = trn[3][idx]; kdb = np.where(full < -50, full, kdb)                 # strength as it happened in this stretch
        tk, tt, tok = roomnet.targets(kdb, trn[4][idx]); ok = trn[5][idx] & (S.max(1) > 10 ** ((trn[0][idx][:, 0, sl].max((1, 2))[:, None] - 40.0) / 10))
        k, t = model(X); mk = torch.from_numpy(ok.astype(np.float32)); mt = torch.from_numpy((ok & tok & (kdb > -25.0)).astype(np.float32))
        l = ((k - torch.from_numpy(tk.astype(np.float32))).abs() * mk).sum() / mk.sum().clamp(min=1.0) + 2.0 * ((t - torch.from_numpy(tt.astype(np.float32))).abs() * mt).sum() / mt.sum().clamp(min=1.0)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step(); tot += float(l.detach())
    r, K, Tt = evaluate(); sc = r['dev'][0] / 3.0 + np.log2(r['dev'][3]) / np.log2(1.2)
    print('epoch %2d  loss %.4f  %.0f s | ' % (ep, tot / (N // BS), time.time() - te) + ' | '.join('%s: strength |err| %.1f dB (per sound: bias %+.1f, |err| %.1f), decay time x%.2f off (per sound within 20%%: %.0f%%, bias x%.2f)' % ((lab,) + r[sp]) for lab, sp in (('dev', 'dev'), ('unseen', 'test'))), flush=True)
    if sc < best: best = sc; torch.save(dict(cfg=model.cfg, state=model.state_dict(), kind='roomnet'), '/tmp/claude-0/v2/train/%s.pt' % TAG); np.savez('/tmp/claude-0/v2/train/%s.sb.npz' % TAG, K=K, T=Tt)
print('done, %.0f s' % (time.time() - t0))
