# The exciter cases as they are: by how many dB each outside exciter lifted each band of sound A and sound B (the truth),
# in a few broad bands, and how the lift differs between quiet and loud frames.
import sys, os, numpy as np
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import truth, ffeat, common
roots = ['/tmp/claude-0/v2/bench/sets/exciter', '/tmp/claude-0/v2/four/sets2/exciter']
FC = ffeat.FC; edges = [50, 500, 2000, 4000, 8000, 12000, 20000]
print('%-22s %-10s' % ('case', 'effect'), ' '.join('%6s' % ('<%dk' % (e // 1000) if e >= 1000 else '<%d' % e) for e in edges[1:]), '  quiet-loud (4k+)')
for r in roots:
    for name in sorted(os.listdir(r)):
        d = os.path.join(r, name) + '/'
        for which in 'AB':
            c = truth.load(d, which); Ed = ffeat.energies(c['dry']); Ew = ffeat.energies(c['wet']); n = min(len(Ed), len(Ew)); Ed, Ew = Ed[:n], Ew[:n]
            L = 10 * np.log10((Ew.mean(0) + 1e-30) / (Ed.mean(0) + 1e-30)); row = []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (FC >= a) & (FC < b); row.append(10 * np.log10((Ew[:, m].sum() + 1e-30) / (Ed[:, m].sum() + 1e-30)))
            hi = FC >= 4000; lev = Ed[:, hi].sum(1); q = lev <= np.percentile(lev, 30); l = lev >= np.percentile(lev, 70)
            dq = 10 * np.log10((Ew[q][:, hi].sum() + 1e-30) / (Ed[q][:, hi].sum() + 1e-30)); dl = 10 * np.log10((Ew[l][:, hi].sum() + 1e-30) / (Ed[l][:, hi].sum() + 1e-30))
            print('%-22s %-10s' % (name + ' ' + which, str(c['meta']['effect'])[:10]), ' '.join('%6.1f' % v for v in row), '  %5.1f %5.1f' % (dq, dl))
