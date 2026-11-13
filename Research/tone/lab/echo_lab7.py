# Why are simulated reflections missed? Look at the true delay directly.
import sys, os, glob, json
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
import feat, echo2
files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[2::9]; r = np.random.default_rng(3); reasons = {}; n = 0
for p in files:
    x = sf.read(p, always_2d=True)[0].astype(np.float64); P = feat.frames(x[:, :2]); act = feat.active(P); Q0 = (P[0] + P[1])[act]
    d = float(np.exp(r.uniform(np.log(0.6 * 48), np.log(12 * 48)))); g0 = 0.45 * float(r.choice([-1, 1])); Q = Q0 * np.exp(2 * echo2.comb_ln(d, g0))[None, :]; f = echo2.find3(Q); n += 1
    if f and abs(f[0][0] - d) < 0.6: reasons['found'] = reasons.get('found', 0) + 1; continue
    T = len(Q); nseg = min(6, T // 12); parts = [Q] + [Q[i * T // nseg:(i + 1) * T // nseg] for i in range(nseg)]; LM = []; WW = []
    for pp in parts: Lm, lev = echo2.lmean_ln(pp); LM.append(Lm); WW.append(echo2.weight(lev))
    kind = os.path.basename(p).split('_')[1]
    if WW[0].sum() < 50: why = 'band too narrow'
    else:
        g = float(echo2.project(LM[0], WW[0], d)[0]); g2 = float(echo2.project(LM[0], WW[0], 2 * d)[0]); gs = [float(echo2.project(LM[i], WW[i], d)[0]) if WW[i].sum() >= 50 else 0.0 for i in range(1, nseg + 1)]
        c = np.abs(echo2.cepstrum(LM[0], WW[0])); k = int(round(d)); rank = int((c[echo2.QLO:echo2.QHI] > c[k - 1:k + 2].max()).sum())
        why = []
        if abs(g) < (0.3 if d < 48 else 0.2): why.append('gain too small')
        if g2 > -0.1 * g * g: why.append('second peak')
        if not all(np.sign(v) == np.sign(g) and abs(v) >= 0.4 * abs(g) for v in gs): why.append('sixths')
        if any(w.sum() < 50 for w in WW[1:]): why.append('(a sixth too narrow)')
        if f: why.append('other first')
        why = ' + '.join(why) or 'passes at true delay (rank %d)' % rank
        print('%-16s d %.1f g0 %+.2f: g %+.2f g2 %+.3f (need <= %+.3f) sixths %s | weight %.0f | found %s | %s' % (kind, d, g0, g, g2, -0.1 * g * g, ' '.join('%+.2f' % v for v in gs), WW[0].sum(), f[:1], why))
    reasons[why if isinstance(why, str) else 'x'] = reasons.get(why, 0) + 1
print(n, reasons)
