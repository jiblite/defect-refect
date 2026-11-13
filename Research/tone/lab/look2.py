# How exactly do the long-term statistics carry the curve (stat(wet) - stat(dry) vs the true curve), and how big is the dry sound's own fine detail?
import sys, time
import numpy as np
from base import *
import fir, ltas
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
grid = fir.log_grid(24); f = np.fft.rfftfreq(N, 1 / SR)
keys = ['pmean', 'lmean', 'p50', 'p75', 'p90']
acc = {}
for d in case_dirs():
    c = load(d, 'A'); m = c['meta']
    if m['split'] != 'dev': continue
    ft, tdb = true_db(c); ct = cell_mean(ft, tdb.mean(0), grid)
    Pd = ltas.frames_db(c['dry'], N); Pw = ltas.frames_db(c['wet'], N)
    sd = ltas.stats(Pd); sw = ltas.stats(Pw)
    # bands that matter: dry long-term level within 50 dB of its loudest band
    ld = ltas.to_grid(f, sd['pmean'][0], grid); aud = ld > ld.max() - 50
    s = '%-22s' % c['name']
    for k in keys:
        gd = ltas.to_grid(f, sd[k][0], grid); gw = ltas.to_grid(f, sw[k][0], grid); e = (gw - gd) - ct; e -= e[aud].mean()
        fine_d = gd - ltas.gsmooth(gd, 12); fine_c = ct - ltas.gsmooth(ct, 12)      # fine = minus a half-octave-sigma Gaussian trend
        acc.setdefault(k, []).append((m['effect'], np.sqrt((e[aud] ** 2).mean()), np.sqrt((fine_d[aud] ** 2).mean()), np.sqrt((fine_c[aud] ** 2).mean())))
        s += ' | %s err %4.2f dryfine %4.2f' % (k, acc[k][-1][1], acc[k][-1][2])
    s += ' | curvefine %4.2f curve rms %4.2f' % (acc['lmean'][-1][3], np.sqrt(((ct - ct[aud].mean())[aud] ** 2).mean()))
    print(s)
print('means over dev cases (N=%d): equivariance error rms dB / dry fine rms dB' % N)
for k in keys: print(k, '%.2f / %.2f' % (np.mean([a[1] for a in acc[k]]), np.mean([a[2] for a in acc[k]])))
for e in sorted(set(a[0] for a in acc['lmean'])): print(e, ' '.join('%s %.2f/%.2f' % (k, np.mean([a[1] for a in acc[k] if a[0] == e]), np.mean([a[2] for a in acc[k] if a[0] == e])) for k in keys))
