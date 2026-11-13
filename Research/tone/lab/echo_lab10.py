import sys, os, glob
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
import feat, echo2
files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[1::3][:150]; shown = 0
for i, p in enumerate(files):
    x = sf.read(p, always_2d=True)[0].astype(np.float64); P = feat.frames(x[:, :2]); act = feat.active(P); Q = (P[0] + P[1])[act]; r = np.random.default_rng(i)
    for g in (0.3, 0.45, 0.7):
        d = float(np.exp(r.uniform(np.log(0.6 * 48), np.log(12 * 48)))); sg = g * float(r.choice([-1, 1]))
        if g != 0.7: continue
        Qc = Q * np.exp(2 * echo2.comb_ln(d, sg))[None, :]; f = echo2.find4(Qc)
        if f and abs(f[0][0] - d) >= 2.0 and shown < 2:
            shown += 1; print('true d %.2f g %+.2f' % (d, sg)); echo2.find4(Qc, verbose=True, max_echoes=1)
            Lm, lev = echo2.lmean_ln(Qc); w = echo2.weight(lev); print('  weight sum %.0f; valid band: %.0f..%.0f Hz' % (w.sum(), feat.FB[np.flatnonzero(w > 0.01)[0]], feat.FB[np.flatnonzero(w > 0.01)[-1]]))
            c = echo2.cepstrum(Lm, w); k = int(round(d)); print('  cepstrum around the true delay:', np.round(c[k - 8:k + 9], 2))
            Lm0, lev0 = echo2.lmean_ln(Q); print('  lmean wet - lmean dry - comb: rms %.3f nepers over the weighted band (0 = the statistic carries the comb exactly)' % np.sqrt((((Lm - Lm0 - echo2.comb_ln(d, sg)) ** 2) * w).sum() / w.sum()))
