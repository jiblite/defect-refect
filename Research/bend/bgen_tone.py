# The bend study: more training examples for the meter, with made steady tones in them. A plain tone's values pile up near
# its top much as a bent sound's do (a sine spends a third of its time within 1 dB of its peak), and the training clips hold
# few sounds of that kind, so a meter taught on them alone takes a sine for a saturated sound. Here: tones by themselves and
# tones mixed with a training clip, as they are (no bend); and the mix, or a tone played as notes of many levels, through a
# box of the family (a bend). A steady tone by itself through a box is left out on purpose: what comes out is another steady
# tone, and nothing in it tells that a box was there.
#   python3 bgen_tone.py N seed out.npz        (a name with "val" uses the held-out tenth of the clips)
import os, sys, glob, time, numpy as np, soundfile as sf, scipy.signal as ss
import bfeat, bchain
from bgen import proxy
SR = 48000
def tone(rng, n, notes=None):
    """A made tone, (n, 2). notes: True = played as notes of many levels, False = held, None = either."""
    LU = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b)))); t = np.arange(n) / SR; u = rng.random()
    def partials(f0):
        """One voice: its partials as (multiple of f0, strength, phase)."""
        v = rng.random()
        if v < 0.25: return [(1.0, 1.0, rng.uniform(0, 2 * np.pi))]
        K = int(rng.integers(2, 41)); a = rng.uniform(0.3, 3.0); zero = rng.random() < 0.3
        return [(k, rng.uniform(0.2, 1.0) / k ** a * (rng.random() > 0.15 or k == 1), 0.0 if zero else rng.uniform(0, 2 * np.pi)) for k in range(1, K + 1) if k * f0 < 20000]
    def voice(f0, bend_hz=None):
        ph = 2 * np.pi * f0 * t if bend_hz is None else 2 * np.pi * np.cumsum(bend_hz) / SR
        if vib: ph = ph + vib * f0 / vr * np.sin(2 * np.pi * vr * t)
        return sum(a * np.sin(k * ph + p) for k, a, p in partials(f0))
    vib = rng.uniform(0.002, 0.02) if rng.random() < 0.4 else 0.0; vr = rng.uniform(3.0, 7.5); f0 = LU(40, 3000)
    if notes is None: notes = rng.random() < 0.35
    if u < 0.45: y = voice(f0)
    elif u < 0.60: y = sum(voice(f0 * r) * rng.uniform(0.4, 1.0) for r in [1.0] + list(2 ** (rng.integers(1, 20, int(rng.integers(1, 5))) / 12.0)))
    elif u < 0.70: y = voice(f0) + voice(f0 * (1 + rng.uniform(0.001, 0.03))) * rng.uniform(0.5, 1.0)
    elif u < 0.82: y = np.sin(2 * np.pi * f0 * t + rng.uniform(0.2, 4.0) * np.sin(2 * np.pi * f0 * rng.choice([0.5, 1.0, 2.0, 3.0, 1.41, 0.01]) * t))
    else:                                                                                  # struck: partials at any pitches, each dying away at its own rate, struck again and again
        y = np.zeros(n); P = [(LU(60, 8000), rng.uniform(0.2, 1.0), LU(0.05, 2.0)) for _ in range(int(rng.integers(2, 13)))]; at = 0.0
        while at < n / SR:
            i0 = int(at * SR); tt = t[:n - i0]; g = rng.uniform(0.2, 1.0)
            y[i0:] += g * sum(a * np.exp(-tt / d) * np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.28)) for f, a, d in P); at += LU(0.08, 2.5)
        notes = False
    if notes:                                                                              # cut into notes: each with its own level, a quick rise and a slower fall
        env = np.zeros(n); at = 0.0
        while at < n / SR:
            L = LU(0.08, 1.5); i0 = int(at * SR); m = min(n - i0, int(L * SR))
            if m > 16: tt = np.arange(m) / SR; env[i0:i0 + m] = 10 ** (rng.uniform(-30, 0) / 20) * np.minimum(tt / LU(0.002, 0.05), 1.0) * np.exp(-tt / LU(0.05, 3.0)) * np.minimum((m - np.arange(m)) / (0.005 * SR), 1.0)
            at += L + (rng.uniform(0, 0.3) if rng.random() < 0.3 else 0.0)
        y = y * env
    else:
        if rng.random() < 0.5: y = y * (1 + rng.uniform(0.05, 0.6) * np.sin(2 * np.pi * LU(0.2, 8.0) * t + rng.uniform(0, 6.28)))
        if rng.random() < 0.4: k = int(rng.integers(3, 9)); y = y * 10 ** (np.interp(t, np.linspace(0, t[-1], k), rng.uniform(-20, 0, k)) / 20)
    y = y / (np.abs(y).max() + 1e-12); z = rng.standard_normal((n, 2))
    if rng.random() < 0.5: z = ss.lfilter([1.0], [1.0, -rng.uniform(0.5, 0.99)], z, axis=0)
    z = z / (np.abs(z).max() + 1e-12) * 10 ** (rng.uniform(-70, -18) / 20)
    r = np.stack([y, y if rng.random() < 0.7 else np.roll(y, int(rng.integers(1, 200))) * rng.uniform(0.5, 1.0)], 1); return r + z
def example(rng, files):
    n = int(rng.uniform(2.0, 6.0) * SR); u = rng.random(); mixed = False
    def clip():
        x, sr = sf.read(files[rng.integers(len(files))], dtype='float64', always_2d=True); a = rng.integers(0, max(1, len(x) - n)); x = x[a:a + n]
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        return x
    if u < 0.42: x = tone(rng, n); kind = 'tone'
    elif u < 0.62 or (u >= 0.62 and rng.random() < 0.75):
        c = clip(); m = len(c)
        if np.abs(c).max() < 1e-4: return None
        x = c / np.abs(c).max() + tone(rng, n)[:m] * 10 ** (rng.uniform(-15, 15) / 20); kind = 'tonemix'; mixed = True
    else: x = tone(rng, n, True); kind = 'tone'
    if rng.random() < 0.5: x = -x
    x = x / np.abs(x).max() * 0.5; y = x; ans = np.zeros((2, len(bfeat.GRID)), np.float32); present = 0.0; d = 0.0
    if u >= 0.62:
        s = bchain.draw(rng); y, uu = bchain.run(x, s); kind = s['name']
        if not np.isfinite(y).all() or np.abs(y).max() < 1e-6: return None
        ans = np.clip(bfeat.answer(uu, y), 0.0, 40.0); d = proxy(x, y); present = float(d >= 0.7)
        if not present: kind = 'unheard'
    y = y * (0.5 / (np.abs(y).max() + 1e-12)); X, vp = bfeat.features(y)
    return X, ans, present, kind, d, np.float32(20 * np.log10(vp[0] / vp[1]))
if __name__ == '__main__':
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]; files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/*')); hold = np.arange(len(files)) % 10 == 0
    files = [f for f, h in zip(files, hold) if (h if 'val' in out else not h)]; t0 = time.time(); PART = 1500; made = 0
    for part in range((n + PART - 1) // PART):
        name = '%s_%02d.npz' % (out[:-4], part); m = min(PART, n - part * PART)
        if os.path.exists(name): continue
        rng = np.random.default_rng([seed, part]); X, Y, P, K, D, R = [], [], [], [], [], []
        while len(P) < m:
            r = example(rng, files)
            if r is None: continue
            X.append(r[0]); Y.append(r[1]); P.append(r[2]); K.append(r[3]); D.append(r[4]); R.append(r[5])
        np.savez(name + '.tmp.npz', X=np.stack(X), Y=np.stack(Y), P=np.array(P, np.float32), K=np.array(K), D=np.array(D, np.float32), R=np.array(R, np.float32))
        os.replace(name + '.tmp.npz', name); made += m
        print(name, m, 'examples, present %.2f, %.0f s' % (np.mean(P), time.time() - t0), {k: int((np.array(K) == k).sum()) for k in sorted(set(K))}, flush=True)
    print(out, 'done,', made, 'made now, %.0f s' % (time.time() - t0), flush=True)
