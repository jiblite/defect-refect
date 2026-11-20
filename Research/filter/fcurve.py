# The fixed-filter study: the family of filter curves the reader is taught with (dB at the band centres, 0 dB in the pass
# band), drawn at random: low-pass, high-pass and band-pass of several builds, with and without a resonance.
import numpy as np
SR = 48000
def _ratio(f, fc, digital):
    """Frequency against the corner; digital: as a filter made at 48 kHz bends it (its response is squeezed against half the rate)."""
    return np.tan(np.pi * np.minimum(f, 0.4999 * SR) / SR) / np.tan(np.pi * min(fc, 0.49 * SR) / SR) if digital else f / fc
def _db(p): return 10 * np.log10(np.maximum(p, 1e-30))
def lp2(r, q): return _db(1.0 / ((1 - r * r) ** 2 + (r / q) ** 2))
def butter(r, n): return _db(1.0 / (1 + r ** (2 * n)))
def ladder(r, k, n=4): s = 1j * r; return _db(np.abs((1 + k) / ((1 + s) ** n + k)) ** 2)
def bell(r, g, q): A = 10 ** (g / 40); s = 1j * r; return _db(np.abs((s * s + s * A / q + 1) / (s * s + s / (A * q) + 1)) ** 2)
def bp2(r, q): return _db((r / q) ** 2 / ((1 - r * r) ** 2 + (r / q) ** 2))
def draw(rng, f, kind=None):
    """One curve of the family: (dB at f, description)."""
    LU = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    kind = kind or rng.choice(['low', 'high', 'band'], p=[0.45, 0.40, 0.15]); dig = bool(rng.random() < 0.5)
    if kind == 'band':
        if rng.random() < 0.6:
            fc = LU(150, 8000); q = LU(0.5, 7); r = _ratio(f, fc, dig); L = bp2(r, q) * (2 if rng.random() < 0.35 else 1); d = dict(kind=kind, fc=fc, q=q)
        else:
            f1 = LU(100, 2500); f2 = f1 * LU(2.5, 30); n1 = int(rng.choice([1, 2, 4])); n2 = int(rng.choice([1, 2, 4]))
            L = butter(1.0 / np.maximum(_ratio(f, f1, dig), 1e-9), n1) + butter(_ratio(f, min(f2, 18000), dig), n2); d = dict(kind=kind, fc=float(np.sqrt(f1 * min(f2, 18000))), q=0.0)
        return L - L.max(), d
    fc = LU(150, 16000) if kind == 'low' else LU(60, 6000); r = _ratio(f, fc, dig)
    if kind == 'high': r = 1.0 / np.maximum(r, 1e-9)
    u = rng.random()
    if u < 0.35: q = LU(0.5, 8); L = lp2(r, q); d = dict(build='one', q=q)
    elif u < 0.60: q = LU(0.5, 8); q2 = 0.7071 if rng.random() < 0.6 else LU(0.5, 1.2); k = LU(0.85, 1.18); L = lp2(r, q) + lp2(r * k, q2); d = dict(build='two', q=q)
    elif u < 0.80:
        n = int(rng.choice([1, 2, 3, 4, 6])); L = butter(r, n); d = dict(build='flat%d' % n, q=0.7)
        if rng.random() < 0.5: g = rng.uniform(2, 14); qb = LU(1.5, 8); L = L + bell(r / LU(0.6, 1.0), g, qb); d = dict(build='flat%d+peak' % n, q=qb)
    else: k = rng.uniform(0, 3.4); L = ladder(r, k); d = dict(build='ladder', q=k)
    d.update(kind=kind, fc=fc); return L, d
