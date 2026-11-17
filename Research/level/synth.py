# My own level-movement effects for tuning (never used for judging): a gain per channel that moves over time.
#   make(n, rng, kind) -> (gain (n, 2) as linear factors, params)      kinds: 'trem', 'gate', 'pan'
import numpy as np
SR = 48000
def lfo(ph, shape):
    """One cycle of an LFO, 0..1, at phase ph (turns)."""
    f = ph - np.floor(ph)
    if shape == 'sine': return 0.5 - 0.5 * np.cos(2 * np.pi * f)
    if shape == 'tri': return 1 - np.abs(2 * f - 1)
    if shape == 'sawup': return f
    if shape == 'sawdown': return 1 - f
    if shape == 'sine2': return (0.5 - 0.5 * np.cos(2 * np.pi * f)) ** 2
    raise ValueError(shape)
def smooth_edges(g, ms):
    if ms <= 0: return g
    c = np.exp(-1.0 / (ms * 1e-3 * SR)); import scipy.signal as ss; return ss.lfilter([1 - c], [1, -c], g, zi=[g[0] * c])[0]
def make(n, rng, kind, rate=None):
    t = np.arange(n) / SR; U = rng.uniform; LU = lambda a, b: float(np.exp(U(np.log(a), np.log(b))))
    if kind == 'trem':
        rate = rate or LU(1.5, 12); depth = U(0.3, 1.0); shape = str(rng.choice(['sine', 'sine', 'tri', 'sawup', 'sawdown', 'sine2'])); ph = U(0, 1)
        g = 1 - depth * (1 - lfo(rate * t + ph, shape)); law = 'lin'
        if rng.random() < 0.25: g = 10 ** (-(depth * 30) * (1 - lfo(rate * t + ph, shape)) / 20); law = 'db'       # a tremolo that is even in dB
        return np.stack([g, g], 1), dict(kind=kind, rate=rate, depth=depth, shape=shape, phase=ph, law=law)
    if kind == 'gate':
        rate = rate or LU(2, 16); duty = U(0.2, 0.8); depth = U(0.6, 1.0) if rng.random() < 0.7 else 1.0; ph = U(0, 1); pattern = (int(rng.integers(0, 1 << 16)) | 0x1111) if rng.random() < 0.35 else 0xFFFF
        sp = rate * t + ph; step = np.floor(sp).astype(int) % 16; on = ((pattern >> step) & 1) * ((sp - np.floor(sp)) < duty); sm = float(rng.choice([0.0, 0.0, 1.0, 3.0, 8.0]))
        g = 1 - depth * (1 - smooth_edges(on.astype(float), sm)); return np.stack([g, g], 1), dict(kind=kind, rate=rate, duty=duty, depth=depth, phase=ph, pattern=pattern, smooth=sm)
    if kind == 'pan':
        rate = rate or LU(0.15, 3); depth = U(0.5, 1.0); shape = str(rng.choice(['sine', 'sine', 'tri'])); ph = U(0, 1); law = str(rng.choice(['lin', 'lin', 'pow', 'db']))
        s = 2 * lfo(rate * t + ph, shape) - 1
        if law == 'lin': gl, gr = 1 - depth * (1 - s) / 2, 1 - depth * (1 + s) / 2
        elif law == 'pow': a = (np.pi / 4) * (1 + depth * s); gl, gr = np.sin(a) * np.sqrt(2), np.cos(a) * np.sqrt(2)
        else: gl, gr = 10 ** (depth * 12 * s / 20), 10 ** (-depth * 12 * s / 20)
        return np.stack([gl, gr], 1), dict(kind=kind, rate=rate, depth=depth, shape=shape, phase=ph, law=law)
    raise ValueError(kind)
