# When does each outside effect settle after start? Click responses at several times against the response at 6 s (no silence in front).
import sys, json, glob, os
sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np
from zoo import ZOO, apply, SR
from multiprocessing import Pool
def db(a, b): return 10 * np.log10((a ** 2).sum() / max((b ** 2).sum(), 1e-30) + 1e-30)
def one(d):
    m = json.load(open(d + 'meta.json')); name = os.path.basename(d.rstrip('/')); p = m['params']; eff = m['effect']; n = SR * 9; L = SR * 2
    def click(at, amp=1.0):
        x = np.zeros((n, 2)); x[at] = amp; return apply(eff, x, p, pre=0.0)[max(0, at - 2400):at + L] / amp
    ref = click(6 * SR); out = []
    for t in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        out.append('%4.2fs %6.1f' % (t, db(click(int(t * SR)) - ref, ref)))
    return '%-24s click at: %s | at a tenth of the level %6.1f dB' % (name, '  '.join(out), db(click(6 * SR, 0.1) - ref, ref))
if __name__ == '__main__':
    S = '/tmp/claude-0/v2/bench/sets/' + sys.argv[1] + '/'
    ds = sorted(glob.glob(S + '*_00/')) + sorted(glob.glob(S + '*_05/'))
    with Pool(4) as p:
        for r in p.imap(one, ds): print(r, flush=True)
