# The scoreboard's reverb cases as the model sees them (41 bands), with the exact direct part: for checking the trained model.
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, common as cm, truth
from multiprocessing import Pool
OUT = '/tmp/claude-0/v2/train/sb/'; os.makedirs(OUT, exist_ok=True); corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
def one(a):
    d, w = a; name = os.path.basename(d.rstrip('/')); f = OUT + name + '_' + w + '.npz'
    if os.path.exists(f): return name
    c = truth.load(d, w); direct, rev = truth.parts(c); ft = cm.features(c['wet'], direct); m = c['meta']
    ed = (direct ** 2).sum(); er = (rev ** 2).sum(); info = dict(effect=m['effect'], split=m['split'], kind=corp[m[w]]['kind'], clip=m[w], drr=float(10 * np.log10(max(ed, 1e-20) / max(er, 1e-20))))
    np.savez_compressed(f, info=json.dumps(info), **{k: v.astype(np.float16) for k, v in ft.items()}); return name
if __name__ == '__main__':
    jobs = [(d, w) for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/')) for w in 'AB']
    with Pool(4) as p:
        for k, n in enumerate(p.imap_unordered(one, jobs)):
            if k % 60 == 0: print(k, n, flush=True)
