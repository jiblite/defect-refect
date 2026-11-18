# Is the model's weak answer UNDER the sound a matter of fit or of what it has seen? The band-level score (net.proxy:
# 100 = the true direct part, 0 = like doing nothing) of the first look and of both looks, separately for the sound
# itself (the first 6 s) and for the ring-out after it, on the scoreboard's reverb pictures and on a sample of the
# pictures the model was trained on.
#   NP=4 python3 body_diag.py [per-set sample, default 400]     -> body_diag.json
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
T = '/tmp/claude-0/v2/train/'; FIRST = os.environ.get('FIRST', 'unet_x'); PAIR = os.environ.get('PAIR', 'unet_x+unet_xp2_e5'); BODY = 562      # frames in 6 s
def parts(g, LE, LS):
    top = LE.max(); fl = top - 60.0; ref = np.maximum(LS, fl); a = np.minimum(np.abs(np.maximum(LE + g, fl) - ref), 30.0); b = np.minimum(np.abs(np.maximum(LE, fl) - ref), 30.0); out = []
    for lo, hi in ((0, len(LE)), (0, BODY), (BODY, len(LE))):
        den = b[lo:hi].sum(); out.append((100.0 * (1.0 - a[lo:hi].sum() / den) if den > 1e-9 else None, float(b[lo:hi].mean()) if hi > lo else 0.0, float(a[lo:hi].mean()) if hi > lo else 0.0))
    return out
def one(f):
    import predict
    z = np.load(f)
    if 'LE' not in z.files: return None
    ft = {k: z[k].astype(np.float64) for k in ('LE', 'U64', 'U256')}; LS = z['LS'].astype(np.float64); m = json.loads(str(z['info']))
    if len(ft['LE']) < 800: return None
    r = dict(set=os.path.basename(os.path.dirname(f)), split=m.get('split', 'train'), drr=float(m.get('drr', 99.0)), kind=m.get('kind', ''), fx=m.get('fx', 'none'), engine=m.get('engine', m.get('effect', '')))
    r['first'] = parts(predict.gains(ft, FIRST), ft['LE'], LS); r['pair'] = parts(predict.gains(ft, PAIR), ft['LE'], LS); return r
def show(rows):
    def line(lab, rs):
        if len(rs) < 5: return
        s = '%-34s n=%4d' % (lab, len(rs))
        for look in ('first', 'pair'):
            for p in range(3):
                v = np.array([r[look][p][0] for r in rs if r[look][p][0] is not None]); s += ' | %5.1f %5.1f %3.0f%%' % (np.median(v), v.mean(), 100 * (v < 0).mean())
        b = np.array([[r['pair'][p][1] for p in range(3)] for r in rs]); s += ' | untouched dB/cell %.2f %.2f %.2f' % tuple(np.median(b, 0)); print(s)
    print('%-41s | first look: whole, sound itself, ring-out (median mean worse) | both looks: the same' % '')
    for st in sorted(set(r['set'] for r in rows)):
        R = [r for r in rows if r['set'] == st and r['drr'] < 90]
        for sp in sorted(set(r['split'] for r in R)):
            Q = [r for r in R if r['split'] == sp]; line('%s %s' % (st, sp), Q); line('   heavy (direct < 5 dB over)', [r for r in Q if r['drr'] < 5]); line('   light (5 dB and more)', [r for r in Q if r['drr'] >= 5]); line('   very light (12 dB and more)', [r for r in Q if r['drr'] >= 12])
            if st == 'cases_x': line('   no other effect', [r for r in Q if r['fx'] == 'none']); line('   with another effect', [r for r in Q if r['fx'] != 'none'])
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--show': show(json.load(open(T + 'body_diag.json'))); sys.exit(0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400; t0 = time.time(); fs = sorted(glob.glob(T + 'sb/*.npz')); rng = np.random.default_rng(3)
    for c in ('cases', 'cases_w', 'cases_s', 'cases_x'):
        a = sorted(glob.glob(T + c + '/*.npz')); fs += [a[i] for i in sorted(rng.permutation(len(a))[:n])]
    with Pool(int(os.environ.get('NP', '4'))) as p: rows = [r for r in p.map(one, fs, chunksize=4) if r]
    json.dump(rows, open(T + 'body_diag.json', 'w')); show(rows); print('%.0f s' % (time.time() - t0))
