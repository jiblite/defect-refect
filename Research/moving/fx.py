# Moving filters for the fit data (this study's own; the scoreboard's effects are all from outside tools): a state-variable
# filter whose cutoff moves (low-pass, band-pass, high-pass, notch; 12 or 24 dB an octave), a phaser (a chain of first-order
# all-pass stages with feedback), and the ways a cutoff can move (LFO shapes, a smooth random walk, one slow ramp, the
# sound's own loudness, steps).
import numpy as np, numba as nb
SR = 48000

@nb.njit(cache=True)
def svf(x, fc, q, mode, stages, q2):
    """x through a state-variable filter whose cutoff fc (Hz, per sample) moves. mode 0 low-pass, 1 band-pass (peak at 1),
    2 high-pass, 3 notch. stages 1 or 2; the second stage has the damping q2."""
    n = len(x); y = np.empty(n); a1 = 0.0; a2 = 0.0; b1 = 0.0; b2 = 0.0
    k1 = 1.0 / q; k2 = 1.0 / q2
    for i in range(n):
        g = np.tan(np.pi * min(max(fc[i], 20.0), 0.45 * SR) / SR)
        d = 1.0 / (1.0 + g * (g + k1)); v3 = x[i] - a2; v1 = d * a1 + g * d * v3; v2 = a2 + g * d * a1 + g * g * d * v3; a1 = 2.0 * v1 - a1; a2 = 2.0 * v2 - a2
        if mode == 0: o = v2
        elif mode == 1: o = k1 * v1
        elif mode == 2: o = x[i] - k1 * v1 - v2
        else: o = x[i] - k1 * v1
        if stages == 2:
            d = 1.0 / (1.0 + g * (g + k2)); v3 = o - b2; v1 = d * b1 + g * d * v3; v2 = b2 + g * d * b1 + g * g * d * v3; b1 = 2.0 * v1 - b1; b2 = 2.0 * v2 - b2
            if mode == 0: o = v2
            elif mode == 1: o = k2 * v1
            elif mode == 2: o = o - k2 * v1 - v2
            else: o = o - k2 * v1
        y[i] = o
    return y

@nb.njit(cache=True)
def phaser(x, fc, mult, fbk, dry, wet):
    """x through a chain of first-order all-pass stages (stage k turns at fc * mult[k]) with feedback fbk round the chain,
    mixed as dry * x + wet * chain."""
    n = len(x); m = len(mult); y = np.empty(n); s = np.zeros(m); last = 0.0
    for i in range(n):
        v = x[i] + fbk * last
        for k in range(m):
            t = np.tan(np.pi * min(max(fc[i] * mult[k], 20.0), 0.45 * SR) / SR); a = (t - 1.0) / (t + 1.0); o = a * v + s[k]; s[k] = v - a * o; v = o
        last = v; y[i] = dry * x[i] + wet * v
    return y

def lfo(shape, ph):
    """One of the LFO shapes at the phases ph (in turns): -1 .. 1."""
    p = ph % 1.0
    if shape == 'sine': return np.sin(2 * np.pi * p)
    if shape == 'tri': return 2.0 * np.abs(2.0 * p - 1.0) - 1.0
    if shape == 'up': return 2.0 * p - 1.0
    if shape == 'down': return 1.0 - 2.0 * p
    raise ValueError(shape)

def path(r, n, x=None):
    """A way for a cutoff to move over n samples: (octaves about its middle per sample, between -0.5 and 0.5 before the
    reach is put on; a short description)."""
    t = np.arange(n) / SR; kind = r.choice(['lfo', 'lfo', 'lfo', 'lfo', 'lfo', 'walk', 'walk', 'ramp', 'ramp', 'env', 'env', 'steps'])
    if kind == 'env' and x is None: kind = 'lfo'
    if kind == 'lfo':
        rate = float(np.exp(r.uniform(np.log(0.06), np.log(5.0)))); shape = str(r.choice(['sine', 'sine', 'tri', 'tri', 'up', 'down'])); s = 0.5 * lfo(shape, rate * t + r.uniform(0, 1))
        if shape in ('up', 'down'):                       # the jump back takes a few milliseconds
            k = int(SR * 0.004); s = np.convolve(np.pad(s, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), mode='valid')
        return s, dict(kind=kind, rate=rate, shape=shape)
    if kind == 'walk':                                    # smooth and aimless: noise through a slow low-pass, stretched to the full reach
        bw = float(np.exp(r.uniform(np.log(0.15), np.log(2.5)))); m = max(8, int(SR / (bw * 8))); k = n // m + 8; v = r.normal(size=k)
        import scipy.ndimage as ndi
        v = ndi.gaussian_filter1d(v, 8.0 / (2 * np.pi) * 1.3, mode='reflect'); s = np.interp(np.arange(n) / m, np.arange(k), v); s = (s - s.min()) / max(np.ptp(s), 1e-9) - 0.5
        return s, dict(kind=kind, bw=bw)
    if kind == 'ramp':                                    # one slow move: up, down, or there and back
        way = str(r.choice(['up', 'down', 'hump', 'dip'])); u = np.linspace(0.0, 1.0, n); c = float(r.uniform(0.6, 1.6)); u = u ** c
        s = {'up': u - 0.5, 'down': 0.5 - u, 'hump': 0.5 - np.abs(2 * u - 1), 'dip': np.abs(2 * u - 1) - 0.5}[way]
        return s, dict(kind=kind, way=way)
    if kind == 'env':                                     # follows the sound's own loudness (an envelope filter)
        e = np.abs(x); att = np.exp(-1.0 / (SR * r.uniform(0.002, 0.02))); rel = np.exp(-1.0 / (SR * float(np.exp(r.uniform(np.log(0.04), np.log(0.4)))))); f = _follow(e, att, rel)
        f = f / max(np.percentile(f, 99), 1e-9); s = np.clip(f, 0.0, 1.0) ** float(r.uniform(0.4, 1.0)) - 0.5
        if r.random() < 0.25: s = -s
        return s, dict(kind=kind)
    rate = float(np.exp(r.uniform(np.log(1.0), np.log(8.0)))); m = int(SR / rate); k = n // m + 2; v = r.uniform(-0.5, 0.5, k); s = np.repeat(v, m)[:n]
    g = int(SR * 0.005); s = np.convolve(np.pad(s, g, mode='edge'), np.ones(2 * g + 1) / (2 * g + 1), mode='valid')
    return s, dict(kind='steps', rate=rate)

@nb.njit(cache=True)
def _follow(e, att, rel):
    y = np.empty(len(e)); v = 0.0
    for i in range(len(e)):
        c = att if e[i] > v else rel; v = c * v + (1.0 - c) * e[i]; y[i] = v
    return y

def sweep(r, x):
    """A random moving filter: returns (function that runs it on a mono signal, description)."""
    n = len(x); s, d = path(r, n, x); reach = float(r.uniform(0.7, 5.0)); mid = float(np.exp(r.uniform(np.log(250.0), np.log(4000.0))))
    lo = mid * 2.0 ** (-reach / 2); hi = mid * 2.0 ** (reach / 2)
    if lo < 60.0: k = 60.0 / lo; lo *= k; hi *= k
    if hi > 14000.0: hi = 14000.0
    fc = lo * (hi / lo) ** (s + 0.5); mode = int(r.choice([0, 0, 0, 1, 1, 2, 3])); stages = int(r.choice([1, 2])); q = float(np.exp(r.uniform(np.log(0.6), np.log(8.0))))
    q2 = float(r.choice([0.707, q])); mix = 1.0 if r.random() < 0.7 else float(r.uniform(0.5, 0.95))
    if mode == 3: q = float(np.exp(r.uniform(np.log(0.5), np.log(4.0))))
    d.update(fx='sweep', mode=mode, stages=stages, q=q, q2=q2, mix=mix, lo=lo, hi=hi)
    def run(v): return (1.0 - mix) * v + mix * svf(np.ascontiguousarray(v), fc, q, mode, stages, q2)
    return run, d

def phase(r, x):
    """A random phaser: returns (function that runs it on a mono signal, description)."""
    n = len(x); s, d = path(r, n, x); reach = float(r.uniform(0.7, 4.5)); mid = float(np.exp(r.uniform(np.log(200.0), np.log(3500.0))))
    lo = mid * 2.0 ** (-reach / 2); hi = mid * 2.0 ** (reach / 2)
    if lo < 40.0: k = 40.0 / lo; lo *= k; hi *= k
    if hi > 14000.0: hi = 14000.0
    if r.random() < 0.3: fc = lo + (hi - lo) * (s + 0.5)                     # some phasers move evenly in Hz, most in octaves
    else: fc = lo * (hi / lo) ** (s + 0.5)
    m = int(r.choice([2, 4, 4, 6, 6, 8, 10, 12])); sp = 1.0 if r.random() < 0.5 else float(np.exp(r.uniform(np.log(1.05), np.log(1.6))))
    mult = sp ** (np.arange(m) - (m - 1) / 2.0); fbk = 0.0 if r.random() < 0.3 else float(r.uniform(-0.6, 0.8)); wet = float(r.uniform(0.35, 1.0)) * (1.0 if r.random() < 0.85 else -1.0)
    dry = 1.0 if r.random() < 0.5 else 1.0 / (1.0 + abs(wet)); wet = wet * dry                 # dry at full level with the chain on top, or the two shared out
    d.update(fx='phaser', stages=m, spread=sp, fbk=fbk, dry=dry, wet=wet, lo=lo, hi=hi)
    def run(v): return phaser(np.ascontiguousarray(v), fc, mult, fbk, dry, wet)
    return run, d
