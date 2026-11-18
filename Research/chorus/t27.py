# The measures for telling which is louder (the sound or its copy), on the results saved by t26.py.
import sys, glob, os, numpy as np
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
import engine as E, gains as Gn, prim
def lean(y1, D, lo=1):
    B = prim.bands(y1); A = 0.0; Bq = 0.0
    for b in range(lo, Gn.NB): e = np.abs(B[b]); ed = np.abs(Gn._along(B[b], D)); A += float((e * e * ed).sum()); Bq += float((e * ed * ed).sum())
    return float(np.log(max(A, 1e-300) / max(Bq, 1e-300)))
def uneven(x, lo=1):
    B = prim.bands(x); return float(sum((B[b] ** 4).mean() for b in range(lo, Gn.NB)) / max(sum((B[b] ** 2).mean() ** 2 for b in range(lo, Gn.NB)), 1e-300))
print('%-24s %6s | %7s %7s | %7s %7s %7s | %7s %7s %7s | %6s %6s | votes' % ('case', 'lean', 'along u', 'along o', 'peak w', 'peak u', 'peak o', 'unev w', 'unev u', 'unev o', 'err u', 'err o'))
for p in sorted(glob.glob('/dev/shm/chw/*.npz')):
    z = np.load(p); wet, dry, xu, xo, D = z['wet'], z['dry'], z['xu'], z['xo'], z['D']; al = z['along']
    def err(x): k = float((x * dry).sum() / max((x * x).sum(), 1e-300)); return 10 * np.log10(max(((k * x - dry) ** 2).sum(), 1e-300) / max((dry ** 2).sum(), 1e-300))      # dB, after the best level match
    L = lean(wet, D); pu, po = E.spiky(xu), E.spiky(xo); uu, uo = uneven(xu), uneven(xo); votes = [L > 0.02, al[2] < 0.8 * al[1], po > 1.05 * pu, uo > 1.05 * uu]
    print('%-24s %+6.3f | %7.3f %7.3f | %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f | %6.1f %6.1f | %s %d  (wet err %.1f)' % (os.path.basename(p)[:-4], L, al[1], al[2], E.spiky(wet), pu, po, uneven(wet), uu, uo, err(xu), err(xo), ''.join('x' if v else '.' for v in votes), sum(votes), err(wet)), flush=True)
