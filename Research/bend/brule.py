# Rules for what is shown, on the held-back examples: the share of each kind of example that passes.
import sys, numpy as np, torch, btrain, bengine
Xv, Yv, Pv, Rv, Kv = btrain.load('data/b_val*.npz'); outs = []
for p in sys.argv[1:]:
    d = torch.load(p); net = btrain.Net(d['w']); net.load_state_dict(d['state']); net.eval()
    with torch.no_grad(): outs.append(net(Xv, Rv))
S = torch.stack([o[0] for o in outs]).mean(0).numpy(); lg = torch.stack([o[1] for o in outs]).mean(0).numpy()
top = np.array([max(bengine.tidy(s[0], 6.0)[-1], bengine.tidy(s[1], 6.0)[-1]) for s in S]); ttop = np.array([max(bengine.tidy(s[0], 6.0)[-1], bengine.tidy(s[1], 6.0)[-1]) for s in Yv.numpy()]); pr = Pv.numpy() > 0.5
kinds = [('heard, true top 10 dB and more', pr & (ttop >= 10)), ('heard, true top 5 to 10 dB', pr & (ttop >= 5) & (ttop < 10)), ('heard, true top under 5 dB', pr & (ttop < 5)), ('unheard', (Kv == 'unheard')), ('clean clips', Kv == 'clean'), ('limiter', Kv == 'limiter'), ('spread apart', Kv == 'stretched'), ('steps', Kv == 'steps'), ('made tones', Kv == 'tone'), ('tone and clip mixed', Kv == 'tonemix')]
rules = [('logit 3, top 3', (lg >= 3) & (top >= 3)), ('logit 0, top 3', (lg >= 0) & (top >= 3)), ('logit 0, top 5', (lg >= 0) & (top >= 5)), ('logit -2, top 5', (lg >= -2) & (top >= 5)), ('logit -2, top 4', (lg >= -2) & (top >= 4)), ('logit -6, top 5', (lg >= -6) & (top >= 5)), ('logit -6, top 3', (lg >= -6) & (top >= 3))]
print('%-32s %5s | ' % ('kind', 'n') + ' | '.join('%15s' % n for n, _ in rules))
for k, m in kinds: print('%-32s %5d | ' % (k, m.sum()) + ' | '.join('%13.1f %%' % (100 * (f & m).sum() / max(m.sum(), 1)) for _, f in rules))
