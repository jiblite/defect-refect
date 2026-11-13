import sys, os, glob
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
import feat, echo2
files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[1::3][:150]; n = 0
for i, p in enumerate(files):
    x = sf.read(p, always_2d=True)[0].astype(np.float64); P = feat.frames(x[:, :2]); act = feat.active(P); Q = (P[0] + P[1])[act]; r = np.random.default_rng(i)
    f0 = echo2.find4(Q)
    for g in (0.3, 0.45, 0.7):
        d = float(np.exp(r.uniform(np.log(0.6 * 48), np.log(12 * 48)))); sg = g * float(r.choice([-1, 1]))
        if g != 0.7: continue
        f = echo2.find4(Q * np.exp(2 * echo2.comb_ln(d, sg))[None, :])
        if f and abs(f[0][0] - d) >= 2.0: print('%-16s true d %.2f g %+.2f | found %s | on the clean clip: %s' % (os.path.basename(p).split('_')[1], d, sg, ' '.join('(%.2f, %+.2f)' % t for t in f), ' '.join('(%.2f, %+.2f)' % t for t in f0) or 'nothing'))
