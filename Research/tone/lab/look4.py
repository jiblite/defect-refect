# Echo finder statistics: three cepstra (of the mean log spectrum, median over frames of per-frame cepstra, weakest of several percentile spectra)
# on clean training clips (false alarms) and on the dev comb cases (hits).
import sys, time, json, glob
import numpy as np, soundfile as sf
from base import *
import fir, ltas
N = 8192; f = np.fft.rfftfreq(N, 1 / SR); QM = 1441
band = (f >= 100) & (f <= 16000); W = np.zeros_like(f); W[band] = np.hanning(band.sum())
def ceps_of(LdB):
    L = LdB * (np.log(10) / 20); L = (L - (L * W).sum(-1, keepdims=True) / W.sum()) * W
    return (np.fft.irfft(L, N, axis=-1) * N / W.sum())[..., :QM] * 2      # *2: a ripple g*cos(2 pi f d) in ln-magnitude gives g at d
def three(x):
    P = ltas.frames_db(x, N); a = ltas.active(P); Q = P[:, a, :].sum(0)                     # linked: channels summed
    st = ltas.stats(P); c1 = ceps_of(st['lmean'][0])
    Lt = 10 * np.log10(Q + 1e-8 * Q.max(1, keepdims=True) + 1e-30); ct = ceps_of(Lt); c2 = np.median(ct, 0)
    cs = np.stack([ceps_of(st[k][0]) for k in ('p25', 'p50', 'p75', 'lmean', 'pmean')]); same = (np.sign(cs) == np.sign(cs[0])).all(0); c3 = np.where(same, np.sign(cs[0]) * np.abs(cs).min(0), 0.0)
    return c1, c2, c3
if __name__ == '__main__':
    q = np.arange(QM) / 48.0
    if sys.argv[1] == 'clean':
        files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[::4]; out = []
        for i, p in enumerate(files):
            x = sf.read(p, always_2d=True)[0].astype(np.float64); out.append(np.stack(three(x)))
        np.save(WORK + 'ceps_clean.npy', np.array(out, np.float32)); print(len(out))
    else:
        cl = np.load(WORK + 'ceps_clean.npy')        # (clips, 3, QM)
        print('clean clips: percentiles of |c| per quefrency range (c1 mean-log | c2 median of frames | c3 weakest percentile)')
        for lo, hi in ((0.3, 0.6), (0.6, 1), (1, 2), (2, 4), (4, 8), (8, 15), (15, 30)):
            r = (q >= lo) & (q < hi); s = '%4.1f-%4.1f ms ' % (lo, hi)
            for j in range(3):
                mx = np.abs(cl[:, j, r]).max(1); s += '| max over range: p50 %.3f p90 %.3f p99 %.3f max %.3f ' % (np.percentile(mx, 50), np.percentile(mx, 90), np.percentile(mx, 99), mx.max())
            print(s)
        for d in case_dirs():
            c = load(d, 'A'); m = c['meta']
            if m['effect'] != 'tone_comb': continue
            k = m['params']['ms'] * 48; k0 = int(round(k)); s = '%-14s d %.3f ms (%.2f smp) g %+.2f' % (c['name'], m['params']['ms'], k, m['params']['g'])
            for key in ('wet', 'dry'):
                cc = three(c[key]); s += ' | ' + key + ' at d: ' + ' '.join('%+.3f' % (cc[j][k0 - 1:k0 + 2][np.argmax(np.abs(cc[j][k0 - 1:k0 + 2]))]) for j in range(3))
                s += ' at 2d: ' + ' '.join('%+.3f' % (cc[j][2 * k0 - 2:2 * k0 + 3][np.argmax(np.abs(cc[j][2 * k0 - 2:2 * k0 + 3]))]) for j in range(3))
            print(s)
