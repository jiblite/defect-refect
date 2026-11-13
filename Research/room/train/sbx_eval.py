# The model's band-level score on the scoreboard sounds as they are (sb), at 0.3 of their width (sb_n) and in mono (sb_m).
#   MODEL=<name> python3 sbx_eval.py [sets=sb,sb_n,sb_m]
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, net, predict, roomnum
MODEL = os.environ.get('MODEL', 'unet_b9'); sets = (sys.argv[1] if len(sys.argv) > 1 else 'sb,sb_n,sb_m').split(',')
for s in sets:
    out = []
    for f in sorted(glob.glob('/tmp/claude-0/v2/train/%s/*.npz' % s)):
        z = np.load(f); ft = {k: z[k].astype(np.float64) for k in ('LE', 'U64', 'U256')}; g = predict.gains(ft, MODEL); p, base = net.proxy(g, ft['LE'], z['LS'].astype(np.float64)); m = json.loads(str(z['info']))
        out.append((m['split'], m['drr'], p, roomnum.room_numbers(ft['LE'], g) is not None))
    for sp in ('dev', 'test'):
        v = np.array([o[2] for o in out if o[0] == sp]); d = np.array([o[1] for o in out if o[0] == sp]); fd = [o[3] for o in out if o[0] == sp and o[3] is not None]
        print('%-5s %-6s n=%3d | median %5.1f mean %5.1f below zero %3.0f%% (light %5.1f, wet %5.1f)%s' % (s, 'dev' if sp == 'dev' else 'unseen', len(v), np.median(v), np.mean(v), 100 * np.mean(v < 0), np.median(v[d >= 8]) if (d >= 8).any() else float('nan'), np.median(v[d < 0]) if (d < 0).any() else float('nan'), (' | room found %3.0f%%' % (100 * np.mean(fd))) if fd else ''), flush=True)
