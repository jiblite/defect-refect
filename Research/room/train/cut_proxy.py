# How much does a cut-off ending cost the model? The scoreboard's reverb cases as they ring out and cut off at 6 s, scored
# over the same cells (the first 6 s).   MODEL=unet_w python3 cut_proxy.py
import sys, os, glob, json, numpy as np, torch
sys.path.insert(0, '/tmp/claude-0/v2/train'); import net
torch.set_num_threads(1); MODEL = os.environ.get('MODEL', 'unet_w'); ck = torch.load('/tmp/claude-0/v2/train/%s.pt' % MODEL, map_location='cpu', weights_only=False); M = net.UNet(ch=tuple(ck['cfg']['ch'])); M.load_state_dict(ck['state']); M.eval()
def run(z):
    LE = z['LE'].astype(np.float32)
    with torch.no_grad(): return M(torch.from_numpy(net.inputs(LE, z['U64'].astype(np.float32), z['U256'].astype(np.float32)))[None])[0].numpy().astype(np.float64)
def prox(g, LE, LS, top):
    fl = top - 60.0; ref = np.maximum(LS, fl); a = np.minimum(np.abs(np.maximum(LE + g, fl) - ref), 30.0).sum(); b = np.minimum(np.abs(np.maximum(LE, fl) - ref), 30.0).sum(); return 100.0 * (1.0 - a / max(b, 1e-9))
rows = []; K = 560
for f in sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz')):
    z = np.load(f); zc = np.load(f.replace('/sb/', '/sb_c/')); i = json.loads(str(z['info'])); g = run(z); gc = run(zc)
    LE, LS = z['LE'].astype(np.float64), z['LS'].astype(np.float64); LEc, LSc = zc['LE'].astype(np.float64), zc['LS'].astype(np.float64); top = LE.max()
    rows.append((i['split'], prox(g, LE, LS, top), prox(g[:K], LE[:K], LS[:K], top), prox(gc[:K], LEc[:K], LSc[:K], top), prox(g[K:], LE[K:], LS[K:], top), float(np.mean(np.abs(g[:K] - gc[:K]))), float(np.mean(np.abs(g[K - 100:K] - gc[K - 100:K])))))
for sp in ('dev', 'test'):
    v = np.array([r[1:] for r in rows if r[0] == sp]); f = lambda c: 'median %5.1f mean %6.1f below zero %2.0f%%' % (np.median(v[:, c]), np.mean(v[:, c]), 100 * np.mean(v[:, c] < 0))
    print('%-4s n=%d | whole picture: %s | rings out, first 6 s only: %s | cut off, first 6 s: %s | the tail alone: %s | gains differ by %.2f dB (last second: %.2f)' % (sp, len(v), f(0), f(1), f(2), f(3), v[:, 4].mean(), v[:, 5].mean()))
