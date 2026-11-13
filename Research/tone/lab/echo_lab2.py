# Echo finder, second round: search 0.5..15 ms only, and a reflection must show its own second peak (-g^2/2 at twice the delay).
import numpy as np, json, glob, os, sys
WORK = 'work/'; QM = 1441
R = np.load(WORK + 'echo_clean.npy').astype(np.float32); files = [os.path.basename(p) for p in sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))]; kind = np.array([f.split('_')[1] for f in files])
q = np.arange(QM) / 48.0; lo, hi = int(0.5 * 48), int(15 * 48); perm = np.random.default_rng(123).permutation(len(R)); va = perm[:100]
def steady(mn, mx): return np.where(mn > 0, mn, np.where(mx < 0, mx, 0.0))
def detect(c0, mn, mx, T, frac=0.2, niter=1):
    """Returns hit (bool), k (index) per row. T: threshold array over q."""
    s = steady(mn, mx); r = np.abs(s) / T; r[:, :lo] = 0; r[:, hi:] = 0; k = r.argmax(1); n = len(k); i = np.arange(n); hit = r[i, k] >= 1
    g = c0[i, k]; k2 = np.minimum(2 * k, QM - 2); c2 = np.minimum(np.minimum(c0[i, k2 - 1], c0[i, k2]), c0[i, k2 + 1]); ok2 = c2 <= -frac * g * g
    return hit, hit & ok2, k
base = np.load(WORK + 'echo_base.npy')
def report(T, label, frac=0.2):
    c0 = R[:, 2, 0]; hit, ok, k = detect(c0, R[:, 2, 1], R[:, 2, 2], T, frac)
    s = '%-26s clean: peak above threshold %.1f%%, with second-peak test %.1f%%' % (label, 100 * hit.mean(), 100 * ok.mean())
    s += ' (real %.1f%% melody %.1f%% loops %.1f%% synth %.1f%%)' % tuple(100 * ok[kind == kk].mean() for kk in ('real-recording', 'fluid-melody', 'sample-loop', 'synthetic'))
    rng = np.random.default_rng(0); S1 = R[:, 2]; hr = []
    for g in (0.2, 0.3, 0.45, 0.6, 0.8):
        okk = []
        for rep in range(2):
            for sg in (-1, 1):
                d = np.exp(rng.uniform(np.log(0.6 * 48), np.log(12 * 48), len(S1))).astype(int); add = np.zeros((len(S1), QM), np.float32)
                for i, di in enumerate(d):
                    for m in range(1, 5):
                        if m * di < QM: add[i, m * di] = (-1) ** (m + 1) * (sg * g) ** m / m
                h, o, kk_ = detect(S1[:, 0] + add, S1[:, 1] + add, S1[:, 2] + add, T, frac); okk.append((o & (np.abs(kk_ - d) <= 1)).mean())
        hr.append(np.mean(okk))
    print(s + ' | reflections found: ' + ' '.join('g=%.2f %.0f%%' % (g, 100 * h) for g, h in zip((0.2, 0.3, 0.45, 0.6, 0.8), hr)), flush=True)
for T0 in (0.10, 0.15, 0.20, 0.25): report(np.full(QM, T0), 'fixed %.2f' % T0)
for T0 in (0.15, 0.20): report(np.maximum(T0, 0.5 * base), 'max(%.2f, 0.5 p99)' % T0)
for fr in (0.0, 0.1, 0.3): report(np.full(QM, 0.15), 'fixed 0.15, frac %.1f' % fr, fr)
