# Does looking a third time help (the second-look model run again on its own answer)? Band-level score on the scoreboard sounds.
import sys, glob, json, os
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, torch, common as cm, net, roomnum, predict
torch.set_num_threads(1)
A, B = sys.argv[1], sys.argv[2]; m2 = predict._load(B)[1]; rows = []
for f in sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz')):
    z = np.load(f); LE = z['LE'].astype(np.float32); x4 = net.inputs(LE, z['U64'].astype(np.float32), z['U256'].astype(np.float32)); info = json.loads(str(z['info'])); LS = z['LS'].astype(np.float64); g = predict.gains(dict(LE=LE, U64=z['U64'], U256=z['U256']), A); p = [net.proxy(g, LE.astype(np.float64), LS)[0]]
    for k in range(3):
        x = np.concatenate([x4, roomnum.second_look(LE.astype(np.float64), g.astype(np.float64))])
        with torch.no_grad(): g = m2(torch.from_numpy(x.astype(np.float32))[None])[0].numpy().astype(np.float64)
        p.append(net.proxy(g, LE.astype(np.float64), LS)[0])
    rows.append((info['split'], p))
for sp in ('dev', 'test'):
    a = np.array([r[1] for r in rows if r[0] == sp]); print('%-5s' % sp + ' | '.join('look %d: median %.1f mean %.1f below zero %.0f%%' % (k + 1, np.median(a[:, k]), np.mean(a[:, k]), 100 * np.mean(a[:, k] < 0)) for k in range(a.shape[1])))
