# The fixed-filter study: the reader's training. A small net of 1-D convolutions along the bands: in, the six curves of a sound
# and the band's place; out, the filter's curve band by band and whether a filter is there.
#   python3 ftrain.py out.pt seed channels epochs [g]        reads data/g_tr_*.npz and data/g_val.npz (g: the files' first letter)
import sys, os, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
MAIN = __name__ == '__main__'
NB = 104; CH = int(sys.argv[3]) if MAIN and len(sys.argv) > 3 else 48; NIN = 6; PAT = sys.argv[5] if MAIN and len(sys.argv) > 5 else 'g'
class Net(nn.Module):
    def __init__(s, ch=CH, nin=NIN):
        super().__init__(); s.c = nn.ModuleList([nn.Conv1d(nin + 1, ch, 5, padding=2), nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.Conv1d(ch, ch, 5, padding=8, dilation=4),
                                                 nn.Conv1d(ch, ch, 5, padding=16, dilation=8), nn.Conv1d(ch, ch, 5, padding=2)])
        s.curve = nn.Conv1d(ch, 1, 1); s.there = nn.Linear(2 * ch, 1)
    def forward(s, x):
        pos = torch.linspace(-1, 1, NB).expand(x.shape[0], 1, NB); h = torch.cat([x / 20.0 + 2.0, pos], 1)
        for c in s.c: h = Fn.relu(c(h))
        return s.curve(h)[:, 0] * 20.0, s.there(torch.cat([h.mean(2), h.amax(2)], 1))[:, 0]
def load(pat):
    X, Y, P = [], [], []
    for f in sorted(glob.glob(pat)): z = np.load(f); X.append(z['X']); Y.append(z['Y']); P.append(z['P'])
    return torch.tensor(np.concatenate(X)), torch.tensor(np.concatenate(Y)), torch.tensor(np.concatenate(P))
def loss_fn(net, X, Y, P):
    c, t = net(X); return Fn.huber_loss(c, Y, delta=6.0) + 2.0 * Fn.binary_cross_entropy_with_logits(t, P), c, t
if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get("THREADS", "4"))); torch.manual_seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    out = sys.argv[1]; X, Y, P = load('data/%s_tr_*.npz' % PAT); Xv, Yv, Pv = load('data/%s_val.npz' % PAT); net = Net(); EP = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EP * (len(X) // 256 + 1))
    print('examples', len(X), 'weights', sum(p.numel() for p in net.parameters()), flush=True); t0 = time.time()
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(X)); tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]; opt.zero_grad(); l, _, _ = loss_fn(net, X[b], Y[b], P[b]); l.backward(); opt.step(); sched.step(); tot += float(l) * len(b)
        net.eval()
        with torch.no_grad():
            l, c, t = loss_fn(net, Xv, Yv, Pv); pos = Pv > 0.5; err = (c - Yv).abs()
            print('epoch %2d train %.3f val %.3f | curve off by %.2f dB mean on filtered, %.2f on clean | there: right on %.1f %% of filtered, wrongly on %.1f %% of clean | %.0f s' % (
                ep, tot / len(X), float(l), float(err[pos].mean()), float(err[~pos].mean()), 100 * float((t[pos] > 0).float().mean()), 100 * float((t[~pos] > 0).float().mean()), time.time() - t0), flush=True)
        torch.save(net.state_dict(), out)
