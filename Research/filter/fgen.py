# The fixed-filter study: training examples. A clean clip's band energies (sometimes two clips mixed, a stretch of it, with a
# mild colour of its own), through a filter curve of the family or through none, sometimes with a noise floor added
# afterwards; what the reader sees (4 curves over the bands), the curve it should answer with, and whether a filter is there.
import numpy as np, ffeat, fcurve
FC = ffeat.FC
def colour(rng):
    """A mild colour a source may have by itself (not a filter): a tilt and a few broad swells."""
    x = np.log2(FC / 1000.0); c = rng.uniform(-1.5, 1.5) * x
    for _ in range(rng.integers(0, 4)): c = c + rng.uniform(-6, 6) * np.exp(-0.5 * ((x - rng.uniform(-4, 4)) / rng.uniform(1.0, 2.5)) ** 2)
    return c
def shelf(rng):
    """A shelf or a bell, as an equaliser makes it: a fixed filter, but not one of this family (no roll-off): counts as none."""
    x = np.log2(FC / np.exp(rng.uniform(np.log(100), np.log(10000))))
    if rng.random() < 0.6: return rng.uniform(-15, 15) / (1 + np.exp(-rng.choice([-1, 1]) * x * rng.uniform(1.5, 4)))
    return rng.uniform(-12, 12) * np.exp(-0.5 * (x / rng.uniform(0.15, 1.0)) ** 2)
def comb(rng):
    """One reflection's comb: also not of this family."""
    d = np.exp(rng.uniform(np.log(0.3e-3), np.log(12e-3))); g = rng.uniform(0.3, 0.9) * rng.choice([-1, 1]); return 20 * np.log10(np.abs(1 + g * np.exp(-2j * np.pi * FC * d)))
def example(rng, E, kind=None, positive=None):
    """E: (clips, frames, bands). Returns features (4, bands), target curve (bands), present (0/1), description."""
    i = rng.integers(len(E)); e = E[i].astype(np.float64)
    if rng.random() < 0.3: j = rng.integers(len(E)); e2 = E[j].astype(np.float64); e = e + e2 * (e.sum() / max(e2.sum(), 1e-30)) * 10 ** (rng.uniform(-12, 12) / 10)
    n = rng.integers(40, e.shape[0] + 1); a = rng.integers(0, e.shape[0] - n + 1); e = e[a:a + n]
    e = e * 10 ** (colour(rng) / 10)
    positive = (rng.random() < 0.6) if positive is None else positive; L = np.zeros(len(FC)); d = dict(kind='none')
    if positive: L, d = fcurve.draw(rng, FC, kind)
    else:
        u = rng.random()
        if u < 0.25: e = e * 10 ** (shelf(rng) / 10)
        elif u < 0.35: e = e * 10 ** (comb(rng) / 10)
    dry, ref0 = ffeat.curves(e); w = dry[0] > dry[0].max() - 50.0      # where the clean sound has something to hear
    ew = e * 10 ** (L / 10)
    if rng.random() < 0.5:                                             # a floor added after the filter (the file's own noise): white or pink, far under the sound
        lev = rng.uniform(-105, -60); tilt = rng.choice([0.0, -3.0]); nz = 10 ** ((lev + tilt * np.log2(FC / 1000.0)) / 10) * FC / FC.sum() * len(FC)
        ew = ew + nz * ew.sum(1).mean()
    X, ref = ffeat.curves(ew)
    if positive:
        heard = float((np.minimum(np.abs(L), 30.0) * w).mean())         # about the scoreboard's distance between the clean and the filtered sound
        if heard < 0.5: positive = False; L = np.zeros(len(FC)); d = dict(kind='unheard')       # the filter changed nothing that can be heard
    return X.astype(np.float32), np.clip(L, -60, 24).astype(np.float32), float(positive), d
if __name__ == '__main__':
    import sys, time
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]; Z = np.load('data/train_E.npz'); E = Z['E']
    names = Z['names']; hold = np.array([hash(str(s)) % 10 == 0 for s in names]) if False else (np.arange(len(names)) % 10 == 0)
    E = E[hold] if 'val' in out else E[~hold]; rng = np.random.default_rng(seed); t0 = time.time()
    X = np.zeros((n, 4, len(FC)), np.float32); Y = np.zeros((n, len(FC)), np.float32); P = np.zeros(n, np.float32); K = []
    for k in range(n): X[k], Y[k], P[k], d = example(rng, E); K.append(d['kind'])
    np.savez(out, X=X, Y=Y, P=P, K=np.array(K)); print(out, n, 'examples, present %.2f, %.0f s' % (P.mean(), time.time() - t0), {k: int((np.array(K) == k).sum()) for k in set(K)})
