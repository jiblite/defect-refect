# The exciter study: training examples for a reader of "added highs", in band energies. An exciter takes the sound above a
# corner, drives it through a curve and mixes the result back in: quiet highs come out lifted by a fixed amount, loud highs by
# less (the curve gives way), and new partials appear an octave and more above what was driven. Here that is done to a clip's
# band energies frame by frame. The answer the reader should give: by how many dB each band was lifted on average.
import numpy as np, ffeat
from fgen import colour, shelf, comb
from fgen2 import moving, tail
import fcurve
FC = ffeat.FC; X2 = np.log2(FC / 1000.0); NB = len(FC)
def excite(rng, e):
    """e (frames, bands) -> the energies after an exciter of random settings, and a description."""
    LU = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    f0 = LU(1200, 9000); order = int(rng.choice([1, 2, 4])); hp = 1.0 / (1.0 + (f0 / FC) ** (2 * order))        # the corner the highs are taken above (power response)
    amount = LU(0.25, 4.0); drive = rng.uniform(-25, 15); posts = int(rng.choice([0, 0, 1, 2])); hard = rng.random() < 0.35
    eh = e * hp; lev = eh.sum(1); ref = np.percentile(lev, 95) + 1e-30; z = lev / ref * 10 ** (drive / 10)       # how far each frame drives the curve (1 = at its knee)
    comp = 1.0 / (1.0 + z) if not hard else 1.0 / (1.0 + z) ** 1.5                                               # the curve gives way: the driven highs come out lower as they get louder
    lin = eh * comp[:, None]
    def up(x, k):       # what was driven, moved up by a factor k in frequency (its k-th partials)
        s = int(round(12 * np.log2(k))); o = np.zeros_like(x); o[:, s:] = x[:, :NB - s]; return o
    w2 = rng.uniform(0, 0.6) if rng.random() < 0.5 else 0.0; w3 = rng.uniform(0.1, 1.0); w5 = rng.uniform(0, 0.5)
    bent = (z * comp)[:, None]                                                                                    # partials grow with the drive and level off
    gen = lin + (w2 * up(eh, 2) + w3 * up(eh, 3) + w5 * up(eh, 5)) * comp[:, None] * bent
    if hard: gen = gen + 0.3 * w3 * (up(eh, 7) + up(eh, 9)) * comp[:, None] * bent
    for _ in range(posts): gen = gen * hp
    out = e + amount ** 2 * gen                                                                                   # in power: the made highs add to the sound (their own phase against it is left out)
    if rng.random() < 0.5: out = out + 2 * amount * np.sqrt(e * lin) * rng.uniform(0.3, 1.0)       # the plain part in step with the sound: it adds in amplitude
    return out, dict(f0=f0, amount=amount, drive=drive)
def example(rng, E, positive=None):
    i = rng.integers(len(E)); e = E[i].astype(np.float64)
    if rng.random() < 0.3: j = rng.integers(len(E)); e2 = E[j].astype(np.float64); e = e + e2 * (e.sum() / max(e2.sum(), 1e-30)) * 10 ** (rng.uniform(-12, 12) / 10)
    n = rng.integers(40, e.shape[0] + 1); a = rng.integers(0, e.shape[0] - n + 1); e = e[a:a + n]; T = n
    e = e * 10 ** (colour(rng) / 10)
    if rng.random() < 0.3: lev = rng.uniform(-75, -30); tilt = rng.choice([0.0, -3.0, -6.0]); e = e + 10 ** ((lev + tilt * X2) / 10) * FC / FC.sum() * NB * e.sum(1).mean()
    positive = (rng.random() < 0.5) if positive is None else positive; L = np.zeros(NB); kind = 'none'
    if not positive:
        u = rng.random()
        if u < 0.10: e = e * 10 ** (shelf(rng) / 10); kind = 'shelf'
        elif u < 0.16: e = e * 10 ** (comb(rng) / 10); kind = 'comb'
        elif u < 0.30: e = e * 10 ** (moving(rng, T) / 10); kind = 'moving'
        elif u < 0.40: e = tail(rng, e); kind = 'tail'
        elif u < 0.55: Lf, d = fcurve.draw(rng, FC); e = e * 10 ** (Lf / 10); kind = 'filter'              # a fixed filter is not an exciter
        ew = e
    else:
        dry = e; ew, d = excite(rng, e); kind = 'exciter'
        md = dry.mean(0); mw = ew.mean(0); L = 10 * np.log10((mw + 1e-30) / (md + 1e-30)); aud = 10 * np.log10(mw + 1e-30) > 10 * np.log10(mw.max() + 1e-30) - 60.0; L = np.where(aud, L, 0.0)
        if float((np.minimum(np.abs(L), 30.0)).mean()) < 0.35: positive = False; L = np.zeros(NB); kind = 'unheard'
    if rng.random() < 0.5: lev = rng.uniform(-105, -60); tilt = rng.choice([0.0, -3.0]); ew = ew + 10 ** ((lev + tilt * X2) / 10) * FC / FC.sum() * NB * ew.sum(1).mean()
    X, ref = ffeat.curves6(ew)
    return X.astype(np.float32), np.clip(L, -40, 40).astype(np.float32), float(positive), dict(kind=kind)
if __name__ == '__main__':
    import sys, time
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]; Z = np.load('data/train_E.npz'); E = Z['E']; hold = np.arange(len(E)) % 10 == 0
    E = E[hold] if 'val' in out else E[~hold]; rng = np.random.default_rng(seed); t0 = time.time()
    X = np.zeros((n, 6, NB), np.float32); Y = np.zeros((n, NB), np.float32); P = np.zeros(n, np.float32); K = []
    for k in range(n): X[k], Y[k], P[k], d = example(rng, E); K.append(str(d['kind']))
    np.savez(out, X=X, Y=Y, P=P, K=np.array(K)); print(out, n, 'examples, present %.2f, %.0f s' % (P.mean(), time.time() - t0), {k: int((np.array(K) == k).sum()) for k in sorted(set(K))})
