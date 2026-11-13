# Can a comb (one short reflection) be seen in the cepstrum of the long-term log spectrum? Wet and dry side by side, all dev cases.
import sys, time
import numpy as np
from base import *
import fir, ltas
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192; stat = sys.argv[2] if len(sys.argv) > 2 else 'lmean'
f = np.fft.rfftfreq(N, 1 / SR)
def ceps(v):
    # v: dB per bin (N/2+1). Weighted to 100 Hz..16 kHz with a smooth taper; mean removed. Returns cepstrum (in 'gain' units: a comb 1+g z^-d gives g at d).
    w = np.zeros_like(f); band = (f >= 100) & (f <= 16000); w[band] = np.hanning(band.sum()); L = v * np.log(10) / 20
    L = (L - (L * w).sum() / w.sum()) * w
    c = np.fft.irfft(L, N) * N / w.sum() ; return c           # scaled so that cos(w d) of amplitude g -> g/2... check below
q = np.arange(N // 2 + 1) / SR * 1000
for d in case_dirs():
    c = load(d, 'A'); m = c['meta']
    if m['split'] != 'dev': continue
    out = []
    for key in ('wet', 'dry'):
        P = ltas.frames_db(c[key], N); s = ltas.stats(P)[stat][0]; cc = ceps(s)[:N // 2 + 1]
        rng = (q >= 0.4) & (q <= 15); k = np.argmax(np.abs(cc) * rng); med = np.median(np.abs(cc[rng])); out.append((q[k], cc[k], med, cc))
    s = '%-22s' % c['name']
    for key, o in zip(('wet', 'dry'), out): s += ' | %s peak at %6.3f ms val %+.3f  noise(med|c|) %.4f  ratio %5.1f' % (key, o[0], o[1], o[2], abs(o[1]) / o[2])
    if m['effect'] == 'tone_comb':
        k = int(round(m['params']['ms'] * 48)); s += ' | true %.3f ms g %+.2f: wet val there %+.3f' % (m['params']['ms'], m['params']['g'], out[0][3][max(k - 1, 0):k + 2][np.argmax(np.abs(out[0][3][max(k - 1, 0):k + 2]))])
    print(s)
