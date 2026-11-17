# The ceiling: the true moving comb (oracle.fit, dry sound known) used to take the effect out and to put it on the other sound.
import sys, time, numpy as np, pickle, os
sys.path.insert(0, '.'); from lab import *
import sweep, oracle
fam = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else ''; rows = []; keep = {}
for d in cases(fam):
    if sub not in d: continue
    cA = load(d, 'A'); cB = load(d, 'B'); Fs = {}
    for c, o, tag in ((cA, cB, 'A'), (cB, cA, 'B')):
        t0 = time.time(); F = oracle.find(c['dry'], c['wet']); Fs[tag] = F
        out = sweep.apply(c['wet'], F, -1.0); onto = sweep.apply(o['dry'], F, 1.0)
        r = score.removed(c['dry'], c['wet'], out)[0]; ro = score.removed(o['wet'], o['dry'], onto)[0]; rows.append((r, ro))
        print('%-20s %s out %7.1f onto %7.1f   %s (%.1fs)' % (c['name'], tag, r, ro, ' | '.join('d %.2f..%.2f ms a %.2f g/a %+.2f fb %+.2f fit %.1f dB' % (f['d'].min() / 48, f['d'].max() / 48, f['a'], f['g'], f['fb'], f['err']) for f in F['ch']), time.time() - t0), flush=True)
    # the same course in time on A and B?
    for ch in range(2):
        a, b = Fs['A']['ch'][ch], Fs['B']['ch'][ch]; both = (a['weight'] > 0.02) & (b['weight'] > 0.02)
        if both.sum() > 5: print('%-20s   ch%d: A and B both seen on %d frames, delay differs by median %.2f samples (90%%: %.2f)' % ('', ch, both.sum(), np.median(np.abs(a['d'] - b['d'])[both]), np.percentile(np.abs(a['d'] - b['d'])[both], 90)))
    keep[cA['name']] = {k: [dict(d=f['d'], g=f['g'], fb=f['fb'], a=f['a'], weight=f['weight'], act=f['act']) for f in Fs[k]['ch']] for k in Fs}
v = np.array(rows); print('ceiling out: median %.1f mean %.1f worse %d/%d | onto: median %.1f mean %.1f worse %d/%d' % (np.median(v[:, 0]), v[:, 0].mean(), (v[:, 0] < 0).sum(), len(v), np.median(v[:, 1]), v[:, 1].mean(), (v[:, 1] < 0).sum(), len(v)))
pickle.dump(keep, open('truth_%s%s.pkl' % (fam, sub), 'wb'))
