# Echo finder experiments. 'clean': cepstra of all clean training clips (for thresholds). 'show': thresholds, false alarms, simulated hit
# rates, and what is found on the dev comb and box cases.
import sys, os, glob, json, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
from multiprocessing import Pool
import feat, echo
WORK = '/tmp/claude-0/v2/tone/lab/work/'; QM = echo.QM
def clean_one(p):
    x = sf.read(p, always_2d=True)[0].astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    P = feat.frames(x[:, :2]); a = feat.active(P); out = []
    for Q in (P[0][a], P[1][a], (P[0] + P[1])[a]):
        c, w, L0 = echo.analyse(Q); out.append(np.stack([c[0], c.min(0), c.max(0)]))
    return np.array(out, np.float16)
if __name__ == '__main__':
    if sys.argv[1] == 'clean':
        files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac')); t0 = time.time()
        with Pool(2) as pool: R = pool.map(clean_one, files, chunksize=8)
        np.save(WORK + 'echo_clean.npy', np.array(R)); print(len(R), time.time() - t0)
    if sys.argv[1] == 'show':
        R = np.load(WORK + 'echo_clean.npy').astype(np.float32)          # (clips, 3 views, [all frames, min over parts, max over parts], QM)
        q = np.arange(QM) / 48.0; c0 = R[:, :, 0]; st = np.where(R[:, :, 1] > 0, R[:, :, 1], np.where(R[:, :, 2] < 0, R[:, :, 2], 0.0))       # steady value per clip, view, quefrency
        perm = np.random.default_rng(123).permutation(len(R)); va = perm[:100]; tr = perm[100:]
        print('clean clips, |steady cepstrum|: percentiles over clips of the largest value in each delay range (pooled view) and of the plain all-frames cepstrum')
        for lo, hi in ((0.25, 0.4), (0.4, 0.6), (0.6, 1), (1, 2), (2, 4), (4, 8), (8, 15), (15, 30)):
            r = (q >= lo) & (q < hi); a = np.abs(st[tr, 2][:, r]).max(1); b = np.abs(c0[tr, 2][:, r]).max(1)
            print('  %5.2f-%5.2f ms  steady p50 %.3f p90 %.3f p98 %.3f p99.5 %.3f max %.3f | plain p50 %.3f p90 %.3f p98 %.3f max %.3f' % (lo, hi, *np.percentile(a, [50, 90, 98, 99.5]), a.max(), *np.percentile(b, [50, 90, 98]), b.max()))
        # threshold: a smooth function of the delay, a multiple of the clean clips' 99th percentile at that delay (all three views of the training clips)
        A = np.abs(st[tr].reshape(-1, QM)); p99 = np.percentile(A, 99, axis=0); k = np.ones(49) / 49; base = np.convolve(np.pad(p99, 24, mode='edge'), k, 'valid'); base = np.maximum(base, 0.02)
        qlo = int(round(0.4 * 48))
        def detect(s, c, thr):
            r = np.abs(s) / thr; r[..., :qlo] = 0; kk = r.argmax(-1); hit = np.take_along_axis(r, kk[..., None], -1)[..., 0] >= 1
            return hit, kk
        def pitchlike(c, kk):      # positive peak whose second peak is positive too
            out = np.zeros(len(kk), bool)
            for i, k_ in enumerate(kk):
                if c[i, k_] > 0 and 2 * k_ + 1 < QM and c[i, 2 * k_ - 1:2 * k_ + 2].max() > 0.5 * c[i, k_]: out[i] = True
            return out
        for kap in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            thr = np.maximum(kap * base, 0.0); s = ''
            for nm, idx in (('train clips', tr), ('held-out clips', va)):
                S = st[idx].reshape(-1, QM); C = c0[idx].reshape(-1, QM); hit, kk = detect(S, C, thr); pl = pitchlike(C, kk); s += ' | %s: false alarms %.1f%% (after pitch test %.1f%%)' % (nm, 100 * hit.mean(), 100 * (hit & ~pl).mean())
            # simulated reflections on held-out clips: gain g at a random whole-sample delay 0.6..12 ms
            rng = np.random.default_rng(0); S1 = R[va][:, 2]; hr = []
            for g in (0.2, 0.3, 0.45, 0.6):
                ok = []
                for rep in range(4):
                    for sg in (-1, 1):
                        d = np.exp(rng.uniform(np.log(0.6 * 48), np.log(12 * 48), len(va))).astype(int); add = np.zeros((len(va), QM), np.float32)
                        for i, di in enumerate(d):
                            for m in range(1, 4):
                                if m * di < QM: add[i, m * di] = (-1) ** (m + 1) * (sg * g) ** m / m
                        mn = S1[:, 1] + add; mx = S1[:, 2] + add; ss_ = np.where(mn > 0, mn, np.where(mx < 0, mx, 0.0)); cc = S1[:, 0] + add; hit, kk = detect(ss_, cc, thr); pl = pitchlike(cc, kk)
                        ok.append((hit & ~pl & (np.abs(kk - d) <= 1)).mean())
                hr.append(np.mean(ok))
            print('threshold = %.1f x p99(delay):%s | found (right delay): %s' % (kap, s, ' '.join('g=%.2f %.0f%%' % (g, 100 * h) for g, h in zip((0.2, 0.3, 0.45, 0.6), hr))))
        print('threshold base (p99 of clean clips) at 0.4 0.6 1 2 4 8 15 30 ms:', ' '.join('%.3f' % base[min(int(round(m * 48)), QM - 1)] for m in (0.4, 0.6, 1, 2, 4, 8, 15, 30)))
        np.save(WORK + 'echo_base.npy', base.astype(np.float32))
