# Several meters read as one, on the held-back examples: the mean of their answers against each one alone.
#   python3 bens.py a.pt b.pt ...
import sys, numpy as np, torch, btrain
Xv, Yv, Pv, Rv, Kv = btrain.load('data/b_val*.npz'); pr = Pv > 0.5; neg = torch.tensor(np.isin(Kv, ['clean', 'limiter', 'stretched', 'steps', 'tone', 'tonemix'])); tn = torch.tensor(np.isin(Kv, ['tone', 'tonemix'])); cl = torch.tensor(Kv == 'clean')
def line(name, S, lg):
    S = S.clamp_min(0); e = (S - Yv).abs()
    print('%-10s heard: off by %.2f dB over the levels, %.2f at the peak | logit above 0 / 1.5 / 3: heard %.1f / %.1f / %.1f %% | no bend %.1f / %.1f / %.1f %% | clean clips %.1f / %.1f / %.1f %% | made tones %.1f / %.1f / %.1f %% | unheard %.1f / %.1f / %.1f %%' % ((name, e[pr].mean(), e[pr][:, :, -1].mean()) +
          tuple(100 * (lg[m] > t).float().mean() for m in (pr, neg, cl, tn, ~pr & ~neg) for t in (0, 1.5, 3))))
outs = []
for p in sys.argv[1:]:
    d = torch.load(p); net = btrain.Net(d['w']); net.load_state_dict(d['state']); net.eval()
    with torch.no_grad(): outs.append(net(Xv, Rv))
    line(p.split('/')[-1][:10], *outs[-1])
line('together', torch.stack([o[0] for o in outs]).mean(0), torch.stack([o[1] for o in outs]).mean(0))
