import sys
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, room, truth, dereverb as dv
name = sys.argv[1]; band = int(sys.argv[2]); t0 = float(sys.argv[3]); t1 = float(sys.argv[4])
c = truth.load('/tmp/claude-0/v2/bench/sets/reverb/' + name + '/'); t = truth.room_truth(c); direct, rev = truth.parts(c)
BM = room.band_matrix(dv.NF); bp = lambda x: (lambda X: (np.abs(X[0]) ** 2 + np.abs(X[1]) ** 2) @ BM.T)(dv.fwd(x))
E = bp(c['wet']); Sx = bp(direct); R = bp(rev); tau = t['tau']; kap = t['kappa']
print('band %d = %.0f Hz, true T60 %.2f s, kappa %.3f (DRR %.1f dB), predelay-ish peak at %.1f ms' % (band, room.FC[band], 13.8155 * tau[band], kap[band], -10 * np.log10(kap[band]), 1000 * t['predelay']))
Rh = room.reverb_track(E[:, band], tau[band], kap[band] / tau[band], dv.DT); top = 10 * np.log10(E[:, band].max())
db = lambda v: 10 * np.log10(np.maximum(v, 1e-20)) - top
print('  time   total  direct  reverb | predicted reverb | true fraction  predicted fraction')
for i in range(int(t0 / dv.DT) + 4, int(t1 / dv.DT) + 4, max(1, int(float(sys.argv[5]) / dv.DT)) if len(sys.argv) > 5 else 2):
    print('%6.3f  %6.1f  %6.1f  %6.1f | %6.1f           | %5.2f  %5.2f' % ((i - 4) * dv.DT, db(E[i, band]), db(Sx[i, band]), db(R[i, band]), db(Rh[i]), R[i, band] / max(Sx[i, band] + R[i, band], 1e-30), min(Rh[i] / max(E[i, band], 1e-30), 9.99)))
