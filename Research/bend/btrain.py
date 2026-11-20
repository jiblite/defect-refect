# The bend study: the meter. From what bfeat sees of both half-waves, per half-wave the squeeze at 11 levels, and for the
# sound whether a bend is there to be heard.
#   python3 btrain.py out.pt seed width epochs            reads data/b_tr_*.npz and data/b_val*.npz
import os, sys, glob, time, numpy as np, torch, torch.nn as nn
NF = 184; NG = 11
class Net(nn.Module):
    def __init__(s, w=384, mean=None, std=None):
        super().__init__(); s.f = nn.Sequential(nn.Linear(2 * NF + 1, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w // 2), nn.ReLU(), nn.Linear(w // 2, NG + 1))
        s.register_buffer('mean', torch.zeros(NF) if mean is None else mean); s.register_buffer('std', torch.ones(NF) if std is None else std)
    def forward(s, X, R):
        """X (B, 2, NF), R (B,) the upper half's peak against the lower half's in dB -> squeeze (B, 2, NG) in dB, logit (B,)."""
        Z = (X - s.mean) / s.std; r = (R / 20.0)[:, None]; a = s.f(torch.cat([Z[:, 0], Z[:, 1], r], 1)); b = s.f(torch.cat([Z[:, 1], Z[:, 0], -r], 1))
        return torch.stack([a[:, :NG], b[:, :NG]], 1) * 10.0, a[:, NG] + b[:, NG]
def load(pat):
    X, Y, P, R, K = [], [], [], [], []
    for f in sorted(glob.glob(pat)): d = np.load(f); X.append(d['X']); Y.append(d['Y']); P.append(d['P']); R.append(d['R']); K.append(d['K'])
    return [torch.tensor(np.concatenate(v)) for v in (X, Y, P, R)] + [np.concatenate(K)]
if __name__ == '__main__':
    out = sys.argv[1]; seed = int(sys.argv[2]); W = int(sys.argv[3]); EP = int(sys.argv[4]); torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get('THREADS', '4')))
    X, Y, P, R, K = load('data/b_tr_*.npz'); Xv, Yv, Pv, Rv, Kv = load('data/b_val*.npz'); print('examples', len(X), 'val', len(Xv), 'present %.2f' % P.mean().item(), flush=True)
    net = Net(W, X.reshape(-1, NF).mean(0), X.reshape(-1, NF).std(0).clamp_min(1e-3)); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); B = 512; steps = EP * (len(X) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps); hub = nn.HuberLoss(delta=3.0); bce = nn.BCEWithLogitsLoss(); t0 = time.time()
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(X)); tot = 0.0
        for i in range(len(X) // B):
            j = perm[i * B:(i + 1) * B]; S, lg = net(X[j], R[j]); loss = hub(S, Y[j]) + 1.5 * bce(lg, P[j]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        net.eval()
        with torch.no_grad(): S, lg = net(Xv, Rv)
        S = S.clamp_min(0); e = (S - Yv).abs(); pr = Pv > 0.5; neg = np.isin(Kv, ['clean', 'limiter', 'stretched', 'steps']); negt = torch.tensor(neg)
        print('epoch %2d train %.3f | heard: squeeze off by %.2f dB over the levels, %.2f at the peak | not a bend: %.2f read at the peak | found %.1f %% of the heard, %.1f %% of the unheard, wrongly %.1f %% of no bend | %.0f s' % (ep, tot / (len(X) // B),
              e[pr].mean(), e[pr][:, :, -1].mean(), S[negt][:, :, -1].mean(), 100 * (lg[pr] > 0).float().mean(), 100 * (lg[~pr & ~negt] > 0).float().mean(), 100 * (lg[negt] > 0).float().mean(), time.time() - t0), flush=True)
        torch.save(dict(state=net.state_dict(), w=W), out)
