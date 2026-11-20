# The bend study: training examples for the meter. A training clip (sometimes two mixed, sometimes through a plain filter,
# a stretch of 2 to 6 s), then one of: nothing; a box of the family (bchain.py); and things that are not a bend: a limiter,
# the opposite of a bend (a sound stretched apart by the inverse of a curve), a sound cut down to few steps.
# Per half-wave: what the meter sees, the true answer, and whether a bend is there to be heard.
#   python3 bgen.py N seed out.npz        (a name with "val" uses the held-out tenth of the clips)
import os, sys, glob, time, numpy as np, soundfile as sf, scipy.signal as ss
import bfeat, bchain
SR = 48000
def proxy(x, y):
    """About the scoreboard's distance between two sounds (one time scale, 32 bands)."""
    def pic(v):
        P = 0
        for c in range(v.shape[1]): f, t, Z = ss.stft(v[:, c], SR, 'hann', 1024, 512, boundary=None, padded=False); P = P + np.abs(Z) ** 2
        e = np.geomspace(40, 20000, 33); return np.stack([P[(f >= a) & (f < b)].sum(0) + 1e-30 for a, b in zip(e[:-1], e[1:])])
    PA, PB = pic(x), pic(y); LA = 10 * np.log10(np.maximum(PA, PA.max() * 1e-6)); LB0 = 10 * np.log10(PB); live = LA > LA.min() + 6
    g = np.median((LA - LB0)[live]) if live.any() else 0.0; return float(np.minimum(np.abs(LA - np.maximum(LB0 + g, LA.min())), 30).mean())
def inverse_of(f, top):
    """The opposite of a curve on [0, top] (by table)."""
    u = np.linspace(0, top, 20001); v = f(u); return lambda y: np.sign(y) * np.interp(np.abs(y), v, u)
def example(rng, files):
    LU = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    x, sr = sf.read(files[rng.integers(len(files))], dtype='float64', always_2d=True)
    if rng.random() < 0.3: x2, _ = sf.read(files[rng.integers(len(files))], dtype='float64', always_2d=True); n = min(len(x), len(x2)); x = x[:n] + x2[:n] * 10 ** (rng.uniform(-12, 12) / 20) * (np.abs(x).max() + 1e-9) / (np.abs(x2).max() + 1e-9)
    n = int(rng.uniform(2.0, 6.0) * SR); a = rng.integers(0, max(1, len(x) - n)); x = x[a:a + n]
    if np.abs(x).max() < 1e-4: return None
    if rng.random() < 0.5: x = -x
    if rng.random() < 0.15:                                                                # the sound as it was may be dull or thin by itself
        b, a_ = ss.butter(int(rng.integers(1, 5)), LU(400, 9000) / (SR / 2), 'low') if rng.random() < 0.6 else ss.butter(int(rng.integers(1, 3)), LU(80, 1500) / (SR / 2), 'high'); x = ss.lfilter(b, a_, x, axis=0)
        if np.abs(x).max() < 1e-4: return None
    x = x / np.abs(x).max() * 0.5; u = rng.random(); kind = 'clean'; y = x; ans = np.zeros((2, len(bfeat.GRID)), np.float32); present = 0.0; d = 0.0
    if u < 0.34: pass
    elif u < 0.42: y = bchain.limiter(x, rng); kind = 'limiter'
    elif u < 0.47:
        s = bchain.draw(rng); f = bchain.curve(s['name'], s['p']); D = 10 ** (min(s['drive_db'], 30.0) / 20); top = float(f(np.array(D)))
        y = inverse_of(f, D)(x / np.abs(x).max() * top * 0.999); kind = 'stretched'
    elif u < 0.52:
        q = np.abs(x).max() / 2 ** rng.uniform(1.5, 7.0); y = np.round(x / q) * q; kind = 'steps'
        if rng.random() < 0.5: h = int(rng.integers(2, 9)); y = np.repeat(y[::h], h, axis=0)[:len(y)]
    else:
        s = bchain.draw(rng); y, uu = bchain.run(x, s); kind = s['name']
        if not np.isfinite(y).all() or np.abs(y).max() < 1e-6: return None
        ans = np.clip(bfeat.answer(uu, y), 0.0, 40.0); d = proxy(x, y); present = float(d >= 0.7)
        if not present: kind = 'unheard'
    if not np.isfinite(y).all() or np.abs(y).max() < 1e-6: return None
    y = y * (0.5 / (np.abs(y).max() + 1e-12)); X, vp = bfeat.features(y)
    return X, ans, present, kind, d, np.float32(20 * np.log10(vp[0] / vp[1]))
if __name__ == '__main__':
    # python3 bgen.py N seed out.npz     written in parts of PART examples (out_00.npz, out_01.npz, ...); a part that is there already is not made again
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
