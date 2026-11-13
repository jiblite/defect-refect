# The model cuts too little on wet sounds and too much on lightly reverberant ones. How much would the right overall depth
# per sound be worth, and can it be read off the model's own answer? Depth = b in g' = -30 * sigmoid(logit(-g/30) + b).
import sys, os, glob, json
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, torch, net
MODEL = os.environ.get('MODEL', 'unet_b9'); torch.set_num_threads(1)
ck = torch.load('/tmp/claude-0/v2/train/%s.pt' % MODEL, map_location='cpu', weights_only=False); M = net.UNet(ch=tuple(ck['cfg']['ch'])); M.load_state_dict(ck['state']); M.eval()
BS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
def curve(g, b):
    p = np.clip(-g / 30.0, 1e-6, 1 - 1e-6); return -30.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))
rows = []
for f in sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz')):
    z = np.load(f); LE = z['LE'].astype(np.float32); m = json.loads(str(z['info']))
    with torch.no_grad(): g = M(torch.from_numpy(net.inputs(LE, z['U64'].astype(np.float32), z['U256'].astype(np.float32)))[None])[0].numpy().astype(np.float64)
    LE = LE.astype(np.float64); LS = z['LS'].astype(np.float64); sc = np.array([net.proxy(curve(g, b), LE, LS)[0] for b in BS]); E = 10 ** (LE / 10); G = 10 ** (g / 10); live = LE > LE.max() - 50
    est = 10 * np.log10(max((E * (1 - G))[live].sum(), 1e-30) / max((E * G)[live].sum(), 1e-30))          # the model's own reverb-to-direct ratio, dB
    deep = float(np.mean(g[live] < -10.0)); mean_cut = float(np.mean(g[live]))
    rows.append(dict(split=m['split'], drr=m['drr'], sc=sc, est=est, deep=deep, mean_cut=mean_cut, best=float(BS[np.argmax(sc)])))
for sp in ('dev', 'test'):
    r = [x for x in rows if x['split'] == sp]; s0 = np.array([x['sc'][4] for x in r]); sb = np.array([x['sc'].max() for x in r])
    print('%s n=%d | as is: median %.1f mean %.1f | best depth per sound: median %.1f mean %.1f' % (sp, len(r), np.median(s0), np.mean(s0), np.median(sb), np.mean(sb)))
bb = np.array([x['best'] for x in rows]); 
for k in ('drr', 'est', 'deep', 'mean_cut'):
    v = np.array([x[k] for x in rows]); print('best depth against %-8s: correlation %+.2f' % (k, np.corrcoef(v, bb)[0, 1]))
# a rule: depth = clip(c0 + c1 * est, lo, hi), fitted on the dev sounds by search
dev = [x for x in rows if x['split'] == 'dev']; tst = [x for x in rows if x['split'] == 'test']
def apply(rs, key, c0, c1, lo=-2.0, hi=2.5):
    out = []
    for x in rs:
        b = float(np.clip(c0 + c1 * x[key], lo, hi)); out.append(float(np.interp(b, BS, x['sc'])))
    return np.array(out)
for key in ('est', 'deep', 'mean_cut'):
    v = np.array([x[key] for x in dev]); best = None
    for c1 in np.linspace(-0.6, 0.6, 25) if key != 'deep' else np.linspace(-8, 8, 33):
        for c0 in np.linspace(-3, 3, 49):
            s = apply(dev, key, c0, c1); q = np.median(s) + np.mean(s)
            if best is None or q > best[0]: best = (q, c0, c1)
    _, c0, c1 = best; sd = apply(dev, key, c0, c1); st = apply(tst, key, c0, c1)
    print('rule on %-8s: depth = %.2f %+.3f x | dev median %.1f mean %.1f below zero %.0f%% | unseen median %.1f mean %.1f below zero %.0f%%' % (key, c0, c1, np.median(sd), np.mean(sd), 100 * np.mean(sd < 0), np.median(st), np.mean(st), 100 * np.mean(st < 0)))
np.save('/tmp/claude-0/v2/train/depth_test_%s.npy' % MODEL, np.array([[x['drr'], x['est'], x['deep'], x['mean_cut'], x['best']] + list(x['sc']) for x in rows]))
