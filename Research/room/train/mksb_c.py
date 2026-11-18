# The scoreboard's reverb cases CUT OFF at 6 s (no tail after the sound), as the model sees them, with the exact direct part:
# for checking, while it trains, that the model reads a file that simply stops as well as one that rings out.   -> sb_c/
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, common as cm, truth
from multiprocessing import Pool
OUT = '/tmp/claude-0/v2/train/sb_c/'; OUTE = '/tmp/claude-0/v2/train/sb_e/'; os.makedirs(OUT, exist_ok=True); os.makedirs(OUTE, exist_ok=True); corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); CUT = 6 * 48000
def one(a):
    d, w = a; name = os.path.basename(d.rstrip('/')); c = None
    for out, lo, hi, cut in ((OUT, 0, CUT, 1), (OUTE, 48000, int(5.5 * 48000), 2)):
        f = out + name + '_' + w + '.npz'
        if os.path.exists(f): continue
        if c is None: c = truth.load(d, w); D, R = truth.parts(c)
        direct, rev, wet = D[lo:hi], R[lo:hi], c['wet'][lo:hi]; ft = cm.features(wet, direct); m = c['meta']
        ed = (direct ** 2).sum(); er = (rev ** 2).sum(); info = dict(effect=m['effect'], split=m['split'], kind=corp[m[w]]['kind'], clip=m[w], drr=float(10 * np.log10(max(ed, 1e-20) / max(er, 1e-20))), cut=cut)
        np.savez_compressed(f, info=json.dumps(info), **{k: v.astype(np.float16) for k, v in ft.items()})
    return name
if __name__ == '__main__':
    jobs = [(d, w) for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/')) for w in 'AB']
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        for k, n in enumerate(p.imap_unordered(one, jobs)):
            if k % 80 == 0: print(k, n, flush=True)
