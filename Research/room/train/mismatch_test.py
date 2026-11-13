# Is the model held back because its training pictures were made the old way (FFT smoothing, silence random) and the app's the
# new way (plain sums)? Band-level score of the same sounds with stored (old) pictures and with freshly made (new) ones.
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, torch, common as cm, truth, net, predict, roomnum
torch.set_num_threads(1)
MODELS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['unet_b9', 'unet_b9+unet_p2_e0']
rows = {m: [] for m in MODELS}
for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 3]:
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); ft = cm.features(c['wet'], direct); z = np.load('/tmp/claude-0/v2/train/sb/%s_A.npz' % c['name'])
    old = dict(LE=z['LE'].astype(np.float64), U64=z['U64'].astype(np.float64), U256=z['U256'].astype(np.float64)); LSo = z['LS'].astype(np.float64); T = min(len(old['LE']), len(ft['LE']))
    for m in MODELS:
        gn = predict.gains(ft, m); go = predict.gains(old, m)
        rows[m].append((net.proxy(go, old['LE'], LSo)[0], net.proxy(gn[:T], ft['LE'][:T], ft['LS'][:T])[0], net.proxy(go[:T], ft['LE'][:T], ft['LS'][:T])[0], float(np.mean(np.abs(gn[:T] - go[:T]))), len(ft['LE']), len(old['LE'])))
for m in MODELS:
    a = np.array(rows[m]); print('%-22s old pictures: median %.1f mean %.1f | new pictures: median %.1f mean %.1f | old answer scored on the new pictures: median %.1f mean %.1f | answers differ by %.2f dB on average | frames %d vs %d' % (m, np.median(a[:, 0]), np.mean(a[:, 0]), np.median(a[:, 1]), np.mean(a[:, 1]), np.median(a[:, 2]), np.mean(a[:, 2]), np.mean(a[:, 3]), a[0, 4], a[0, 5]))
