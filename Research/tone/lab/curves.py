# Random tone curves for training. Every curve is computed twice: exactly (dB, cell means on the 1/24-octave grid = what the model has
# to find) and as the analysis sees it (on the FFT bins, with the window's leakage; shifts for the three curve-carrying statistics).
# Types: parametric EQ and resonances (as the dev-split zoo effects sample them), small boxes (dev zoo effect, from a bank), combs,
# and own inventions: 'measured-looking' responses (band limits, slow undulation, bumps, ripple) and tilts.
import numpy as np, scipy.ndimage as ndi, os
import feat
SR = 48000; G = feat.G; SUB = 8; WORK = '/tmp/claude-0/v2/tone/lab/work/'
_e = feat._e; DENSE = np.concatenate([np.exp(np.log(_e[i]) + (np.arange(SUB) + 0.5) / SUB * (np.log(_e[i + 1]) - np.log(_e[i]))) for i in range(G)])
AX = np.concatenate([DENSE, feat.FB]); ND = len(DENSE)              # all curves are evaluated on both axes at once: dense log axis, then the FFT bins
_z1 = np.exp(-2j * np.pi * AX / SR); _z2 = _z1 ** 2
def _bq(b, a): return 20 * np.log10(np.abs((b[0] + b[1] * _z1 + b[2] * _z2) / (a[0] + a[1] * _z1 + a[2] * _z2)) + 1e-12)
def peak(f, g, q):
    A = 10 ** (g / 40); w = 2 * np.pi * min(f, 0.49 * SR) / SR; al = np.sin(w) / (2 * q); c = np.cos(w)
    return _bq((1 + al * A, -2 * c, 1 - al * A), (1 + al / A, -2 * c, 1 - al / A))
def lowshelf(f, g, q=0.707):
    A = 10 ** (g / 40); w = 2 * np.pi * f / SR; al = np.sin(w) / (2 * q); c = np.cos(w); s = 2 * np.sqrt(A) * al
    return _bq((A * ((A + 1) - (A - 1) * c + s), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - s)), ((A + 1) + (A - 1) * c + s, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - s))
def highshelf(f, g, q=0.707):
    A = 10 ** (g / 40); w = 2 * np.pi * f / SR; al = np.sin(w) / (2 * q); c = np.cos(w); s = 2 * np.sqrt(A) * al
    return _bq((A * ((A + 1) + (A - 1) * c + s), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - s)), ((A + 1) - (A - 1) * c + s, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - s))
def comb_cells(ms, g, K=80):
    """Exact cell means of 20 log10 |1 + g e^{-j w d}|, also where a cell holds many comb teeth."""
    d = ms * 1e-3; t1 = 2 * np.pi * _e[:-1] * d; t2 = 2 * np.pi * _e[1:] * d; k = np.arange(1, K + 1)[:, None]
    s = ((-1.0) ** (k + 1) * g ** k / k * (np.sin(k * t2[None]) - np.sin(k * t1[None])) / (k * (t2 - t1)[None])).sum(0); return 20 / np.log(10) * s
def smooth_noise(rng, sigma, rms):
    k = int(4 * sigma) + 1; v = rng.standard_normal(G + 2 * k); t = np.arange(-k, k + 1); w = np.exp(-0.5 * (t / sigma) ** 2); v = np.convolve(v, w, 'valid')[:G]
    return v / (v.std() + 1e-9) * rms
def grid_to_ax(c): return np.interp(np.log(np.maximum(AX, 1.0)), np.log(feat.GRID), c)
U = lambda r, a, b: float(r.uniform(a, b)); LU = lambda r, a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
def pack(v, exact_cells=None):
    """v: dB on AX. Returns (4, G): the exact cell means, and the shifts of lmean, floor and env as the analysis sees them."""
    T = v[:ND].reshape(G, SUB).mean(1) if exact_cells is None else exact_cells
    b = v[ND:]; p = 10 ** (b / 10); pin = p.copy(); pin[1:-1] = p[1:-1] * (2 / 3) + (p[:-2] + p[2:]) / 6; cin = 10 * np.log10(pin + 1e-12)       # the Hann window's leakage into the two neighbours
    sl = feat.to_grid(cin); sf = feat.to_grid(ndi.uniform_filter1d(cin, feat.FW, mode='nearest')); se = feat.to_grid(0.5 * (ndi.uniform_filter1d(cin, feat.EW, mode='nearest') + ndi.maximum_filter1d(cin, feat.EW, mode='nearest')))
    return np.stack([T, sl, sf, se])
def eq(r):
    v = np.zeros(len(AX))
    for _ in range(int(r.integers(1, 5))): v += peak(LU(r, 60, 12000), float(r.choice([-1, 1])) * U(r, 3, 12), LU(r, 0.6, 6))
    if r.random() < 0.5: v += lowshelf(LU(r, 80, 300), U(r, -9, 9))
    if r.random() < 0.5: v += highshelf(LU(r, 3000, 10000), U(r, -9, 9))
    return pack(v)
def res(r):
    v = np.zeros(len(AX)); sgn = -1.0 if r.random() < 0.15 else 1.0
    for _ in range(int(r.integers(2, 7))): v += peak(LU(r, 120, 9000), sgn * U(r, 6, 15), LU(r, 8, 40))
    return pack(v)
def comb(r):
    ms = LU(r, 0.6, 12); g = float(r.choice([-1, 1])) * U(r, 0.3, 0.9); v = 10 * np.log10(1 + g * g + 2 * g * np.cos(2 * np.pi * AX * ms * 1e-3)); return pack(v, comb_cells(ms, g))
def meas(r):
    v = np.zeros(len(AX)); f = np.maximum(AX, 1.0)
    if r.random() < 0.7: fc = LU(r, 30, 250); n = int(r.integers(1, 5)); v += -10 * np.log10(1 + (fc / f) ** (2 * n))
    if r.random() < 0.7: fc = LU(r, 2500, 16000); n = int(r.integers(1, 7)); v += -10 * np.log10(1 + (f / fc) ** (2 * n))
    for _ in range(int(r.integers(0, 9))): v += peak(LU(r, 60, 14000), float(r.choice([-1, 1])) * U(r, 2, 10), LU(r, 1.5, 15))
    c = smooth_noise(r, LU(r, 4, 40), U(r, 1, 5))
    if r.random() < 0.5: c += smooth_noise(r, U(r, 1, 3), U(r, 0.2, 1.5))
    v = v + grid_to_ax(c); return pack(np.maximum(v, v[:ND].max() - 70))
def tilt(r):
    v = U(r, -4, 4) * np.log2(np.maximum(AX, 10.0) / 1000.0)
    if r.random() < 0.5: v = v + (lowshelf(LU(r, 60, 500), U(r, -12, 12)) if r.random() < 0.5 else highshelf(LU(r, 2000, 12000), U(r, -12, 12)))
    return pack(v)
_box = None
def box(r):
    """One channel of a small box from the bank (dev zoo effect): exact cell means from a long FFT, bins from the analysis-size FFT."""
    global _box
    if _box is None: _box = np.load(WORK + 'box_bank.npy')
    h = _box[r.integers(len(_box)), :, r.integers(2)].astype(np.float64); NF = 1 << 15; f = np.fft.rfftfreq(NF, 1 / SR); db = 20 * np.log10(np.abs(np.fft.rfft(h, NF)) + 1e-9)
    df = f[1]; lo = np.clip(np.ceil(_e[:-1] / df).astype(int), 0, len(f)); hi = np.clip(np.ceil(_e[1:] / df).astype(int), 0, len(f)); wide = hi - lo >= 2
    cs = np.concatenate([[0.0], np.cumsum(db)]); T = np.interp(feat.GRID, f, db); T[wide] = (cs[hi[wide]] - cs[lo[wide]]) / (hi[wide] - lo[wide])
    v = np.concatenate([np.zeros(ND), db[::NF // feat.N]]); return pack(v, T)
TYPES = dict(eq=eq, res=res, comb=comb, meas=meas, tilt=tilt, box=box)
MIX = (('eq', .32), ('res', .16), ('meas', .22), ('tilt', .08), ('comb', .06), ('box', .16))
def pool(n, seed, mix=MIX, cache=True):
    """n random curves: (n, 4, G) float32 [exact cell means, lmean shift, floor shift, env shift], type index, type names."""
    p = WORK + 'pool_%d_%d.npz' % (n, seed)
    if cache and os.path.exists(p): z = np.load(p); return z['c'].astype(np.float32), z['ty'], [m[0] for m in mix]
    r = np.random.default_rng(seed); names = [m[0] for m in mix]; pr = np.array([m[1] for m in mix]); pr /= pr.sum(); ty = r.choice(len(names), n, p=pr)
    out = np.stack([TYPES[names[t]](r) for t in ty]).astype(np.float32)
    if cache: np.savez(p, c=out.astype(np.float16), ty=ty)
    return out, ty, names
if __name__ == '__main__':
    import time; t0 = time.time(); c, ty, names = pool(2000, 0, cache=False); print(time.time() - t0, c.shape)
    for k, nm in enumerate(names):
        s = c[ty == k]; print(nm, len(s), 'exact: p5 %.1f p95 %.1f rms %.2f | mean abs (shift - exact): lmean %.2f floor %.2f env %.2f' % (np.percentile(s[:, 0], 5), np.percentile(s[:, 0], 95), (s[:, 0] - s[:, 0].mean(-1, keepdims=True)).std(), np.abs(s[:, 1] - s[:, 0]).mean(), np.abs(s[:, 2] - s[:, 0]).mean(), np.abs(s[:, 3] - s[:, 0]).mean()))
