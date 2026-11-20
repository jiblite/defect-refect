# The fixed-filter study: training examples, second make. As fgen.py, with more of what is NOT a fixed filter: filters that
# move (sweeps, phasers, flangers), a level that moves, a room's tail, a source cut off at a low sample rate; and sources
# with a noise floor of their own, which a filter then shapes. Six curves per sound (ffeat.curves6).
import numpy as np, ffeat, fcurve
from fgen import colour, shelf, comb
FC = ffeat.FC; X2 = np.log2(FC / 1000.0); DT = ffeat.HOP / 48000.0; SHIFT = True
def moving(rng, T):
    """A filter that moves: per frame a curve (T, bands). Not a fixed filter."""
    t = np.arange(T) * DT; rate = np.exp(rng.uniform(np.log(0.1), np.log(3.0))); lfo = np.sin(2 * np.pi * rate * t + rng.uniform(0, 6.28)); u = rng.random()
    if u < 0.45:        # a resonant low-pass or band-pass that sweeps
        mid = np.exp(rng.uniform(np.log(300), np.log(4000))); depth = rng.uniform(1.0, 5.0); q = np.exp(rng.uniform(np.log(1.0), np.log(8.0))); band = rng.random() < 0.4
        fc = mid * 2 ** (0.5 * depth * lfo); r = FC[None, :] / fc[:, None]; return fcurve.bp2(r, q) if band else fcurve.lp2(r, q) * (2 if rng.random() < 0.4 else 1)
    if u < 0.75:        # a phaser's notches
        k = rng.integers(2, 7); c0 = rng.uniform(-2.5, 1.0); sp = rng.uniform(0.5, 1.2); depth = rng.uniform(1.0, 3.0); w = rng.uniform(0.08, 0.25); dep = rng.uniform(6, 24); L = np.zeros((T, len(FC)))
        for i in range(k): L -= dep * np.exp(-0.5 * ((X2[None, :] - (c0 + i * sp + 0.5 * depth * lfo)[:, None]) / w) ** 2)
        return L
    d = np.exp(rng.uniform(np.log(0.5e-3), np.log(6e-3))) * 2 ** (rng.uniform(0.5, 2.0) * lfo); g = rng.uniform(0.4, 0.9) * rng.choice([-1, 1])       # a flanger's moving comb
    return 20 * np.log10(np.abs(1 + g * np.exp(-2j * np.pi * FC[None, :] * d[:, None])) + 1e-6)
def tail(rng, e):
    """A room's tail on the band energies: each band rings on, the highs shorter."""
    T = e.shape[0]; t60 = np.exp(rng.uniform(np.log(0.3), np.log(4.0))) * 2 ** (-rng.uniform(0.0, 0.6) * np.maximum(X2, 0)); k = np.exp(-13.8 * DT / t60); lev = 10 ** (rng.uniform(-12, 6) / 10)
    out = np.zeros_like(e); acc = np.zeros(e.shape[1])
    for i in range(T): acc = acc * k + e[i] * (1 - k); out[i] = e[i] + lev * acc
    return out
def example(rng, E, kind=None, positive=None):
    i = rng.integers(len(E)); e = E[i].astype(np.float64)
    if rng.random() < 0.3: j = rng.integers(len(E)); e2 = E[j].astype(np.float64); e = e + e2 * (e.sum() / max(e2.sum(), 1e-30)) * 10 ** (rng.uniform(-12, 12) / 10)
    n = rng.integers(40, e.shape[0] + 1); a = rng.integers(0, e.shape[0] - n + 1); e = e[a:a + n]; T = n
    if SHIFT and rng.random() < 0.5:                                   # the source played faster or slower: its whole colour moved along the bands
        s = int(rng.integers(-12, 13)); o = np.empty_like(e)
        if s > 0: o[:, s:] = e[:, :len(FC) - s]; o[:, :s] = e[:, :1] * 10 ** (-rng.uniform(0, 12) * (s - np.arange(s))[None, :] / 12 / 10)
        elif s < 0: o[:, :len(FC) + s] = e[:, -s:]; o[:, len(FC) + s:] = e[:, -1:] * 10 ** (-rng.uniform(6, 60) * (1 + np.arange(-s))[None, :] / 12 / 10)
        if s: e = o
    e = e * 10 ** (colour(rng) / 10)
    if rng.random() < 0.3:                                             # the source's own noise floor (a recording's hiss or hum of air), before any filter
        lev = rng.uniform(-75, -30); tilt = rng.choice([0.0, -3.0, -6.0]); e = e + 10 ** ((lev + tilt * X2) / 10) * FC / FC.sum() * len(FC) * e.sum(1).mean()
    positive = (rng.random() < 0.55) if positive is None else positive; L = np.zeros(len(FC)); d = dict(kind='none')
    if not positive:
        u = rng.random()
        if u < 0.12: e = e * 10 ** (shelf(rng) / 10); d = dict(kind='shelf')
        elif u < 0.18: e = e * 10 ** (comb(rng) / 10); d = dict(kind='comb')
        elif u < 0.40: e = e * 10 ** (moving(rng, T) / 10); d = dict(kind='moving')
        elif u < 0.46: e = e * 10 ** (rng.uniform(3, 20) * 0.5 * (np.sin(2 * np.pi * rng.uniform(0.5, 10) * np.arange(T) * DT + rng.uniform(0, 6.28)) - 1) / 10)[:, None]; d = dict(kind='level')
        elif u < 0.58: e = tail(rng, e); d = dict(kind='tail')
        elif u < 0.64: fb = np.exp(rng.uniform(np.log(3400), np.log(16000))); e = e * np.where(FC > fb, 1e-12, 1.0); d = dict(kind='cutoff')
    else:
        L, d = fcurve.draw(rng, FC, kind)
        if rng.random() < 0.08: e = e * 10 ** (rng.uniform(3, 20) * 0.5 * (np.sin(2 * np.pi * rng.uniform(0.5, 10) * np.arange(T) * DT + rng.uniform(0, 6.28)) - 1) / 10)[:, None]        # a filter and a moving level
    dry, ref0 = ffeat.curves6(e); w = dry[0] > dry[0].max() - 50.0
    ew = e * 10 ** (L / 10)
    if rng.random() < 0.5:                                             # a floor added after the filter (the file's own noise): white or pink, far under the sound
        lev = rng.uniform(-105, -60); tilt = rng.choice([0.0, -3.0]); ew = ew + 10 ** ((lev + tilt * X2) / 10) * FC / FC.sum() * len(FC) * ew.sum(1).mean()
    X, ref = ffeat.curves6(ew)
    if positive:
        heard = float((np.minimum(np.abs(L), 30.0) * w).mean())
        if heard < 0.5: positive = False; L = np.zeros(len(FC)); d = dict(kind='unheard')
    return X.astype(np.float32), np.clip(L, -40, 24).astype(np.float32), float(positive), d
if __name__ == '__main__':
    import sys, time
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]; Z = np.load('data/train_E.npz'); E = Z['E']; hold = np.arange(len(E)) % 10 == 0
    E = E[hold] if 'val' in out else E[~hold]; rng = np.random.default_rng(seed); t0 = time.time()
    X = np.zeros((n, 6, len(FC)), np.float32); Y = np.zeros((n, len(FC)), np.float32); P = np.zeros(n, np.float32); K = []
    for k in range(n): X[k], Y[k], P[k], d = example(rng, E); K.append(str(d['kind']))
    np.savez(out, X=X, Y=Y, P=P, K=np.array(K)); print(out, n, 'examples, present %.2f, %.0f s' % (P.mean(), time.time() - t0), {k: int((np.array(K) == k).sum()) for k in sorted(set(K))})
