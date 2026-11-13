# The scoreboard's reverb sounds again as the model sees them, narrowed as a whole: sb_n/ at 0.3 of their width, sb_m/ in mono
# (truth narrowed alike). For checking a model on narrow and mono sound.    python3 mksb_w.py [processes]
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, common as cm, truth
from multiprocessing import Pool
corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); SETS = (('sb_n', 0.3), ('sb_m', 0.0))
for s, _ in SETS: os.makedirs('/tmp/claude-0/v2/train/%s/' % s, exist_ok=True)
def narrowed(x, w): m = x.mean(1); s = 0.5 * (x[:, 0] - x[:, 1]) * w; return np.stack([m + s, m - s], 1)
def one(a):
    d, w = a; name = os.path.basename(d.rstrip('/'))
    if all(os.path.exists('/tmp/claude-0/v2/train/%s/%s_%s.npz' % (s, name, w)) for s, _ in SETS): return name
    c = truth.load(d, w); direct, rev = truth.parts(c); m = c['meta']
    for s, wd in SETS:
        ww, dd = narrowed(c['wet'], wd), narrowed(direct, wd); ed = (dd ** 2).sum(); er = ((ww - dd) ** 2).sum(); ft = cm.features(ww, dd)
        info = dict(effect=m['effect'], split=m['split'], kind=corp[m[w]]['kind'], clip=m[w], drr=float(10 * np.log10(max(ed, 1e-20) / max(er, 1e-20))), width=wd)
        np.savez_compressed('/tmp/claude-0/v2/train/%s/%s_%s.npz' % (s, name, w), info=json.dumps(info), **{k: v.astype(np.float16) for k, v in ft.items()})
    return name
if __name__ == '__main__':
    jobs = [(d, w) for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/')) for w in 'AB']
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        for k, n in enumerate(p.imap_unordered(one, jobs)):
            if k % 60 == 0: print(k, n, flush=True)
