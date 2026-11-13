# Would a fixed "contrast" curve on the model's answer help? g' = -30 * sigmoid(a * logit(-g / 30) + b): a > 1 makes
# decisions firmer, b > 0 moves the middle towards "room". Scored with the band-level proxy on the scoreboard sounds.
import sys, os, glob, json
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, torch, net
MODEL = os.environ.get('MODEL', 'unet_b9'); torch.set_num_threads(1)
ck = torch.load('/tmp/claude-0/v2/train/%s.pt' % MODEL, map_location='cpu', weights_only=False); M = net.UNet(ch=tuple(ck['cfg']['ch'])); M.load_state_dict(ck['state']); M.eval()
rows = []
for f in sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz')):
    z = np.load(f); LE = z['LE'].astype(np.float32)
    with torch.no_grad(): g = M(torch.from_numpy(net.inputs(LE, z['U64'].astype(np.float32), z['U256'].astype(np.float32)))[None])[0].numpy().astype(np.float64)
    rows.append((json.loads(str(z['info'])), LE.astype(np.float64), z['LS'].astype(np.float64), g))
def curve(g, a, b):
    p = np.clip(-g / 30.0, 1e-6, 1 - 1e-6); return -30.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))
def run(a, b):
    out = {}
    for m, LE, LS, g in rows:
        p, base = net.proxy(curve(g, a, b) if (a, b) != (1.0, 0.0) else g, LE, LS); out.setdefault(m['split'], []).append((p, m['drr']))
    return out
print('model', MODEL, '| a b | dev: median mean below0 light wet | unseen: median mean below0 light wet')
best = None
for a in (1.0, 1.25, 1.5, 2.0, 3.0):
    for b in (-0.5, 0.0, 0.5, 1.0, 1.5):
        o = run(a, b); s = []
        for sp in ('dev', 'test'):
            v = np.array([x[0] for x in o[sp]]); d = np.array([x[1] for x in o[sp]]); s.append('%5.1f %5.1f %3.0f%% %5.1f %5.1f' % (np.median(v), np.mean(v), 100 * np.mean(v < 0), np.median(v[d >= 8]), np.median(v[d < 0])))
        print('%.2f %+.1f | %s | %s' % (a, b, s[0], s[1]), flush=True)
