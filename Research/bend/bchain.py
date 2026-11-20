# The bend study: the family of boxes that bend a sound, for the meter's training. A sound goes through a gain (the drive), a
# fixed curve (many shapes, even or uneven for the two halves, with or without plain sound mixed in), and sometimes what such
# boxes have behind the curve: a blocker for the offset, a mild low-pass. Nothing here is any outside tool's code.
import numpy as np, scipy.signal as ss
SR = 48000
def curve(name, p=2.0):
    """An odd curve with slope 1 at 0 (where it has one)."""
    if name == 'pnorm': return lambda x: x / (1 + np.abs(x) ** p) ** (1 / p)
    if name == 'tanh': return np.tanh
    if name == 'hard': return lambda x: np.clip(x, -1, 1)
    if name == 'cubic': return lambda x: np.where(np.abs(x) < 1, x - x ** 3 / 3, np.sign(x) * 2 / 3) * 1.5
    if name == 'atan': return lambda x: np.arctan(x * np.pi / 2) * 2 / np.pi
    if name == 'exp': return lambda x: np.sign(x) * (1 - np.exp(-np.abs(x)))
    if name == 'power': return lambda x: np.sign(x) * ((np.abs(x) + 1.0) ** p - 1.0) / p       # a root-like curve: no ceiling, a line only far under the knee
    raise ValueError(name)
def draw(rng):
    """One box's settings."""
    LU = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    name = rng.choice(['pnorm', 'tanh', 'hard', 'cubic', 'atan', 'exp', 'power'], p=[0.30, 0.18, 0.12, 0.10, 0.08, 0.07, 0.15])
    s = dict(name=str(name), p=LU(1.0, 12.0) if name == 'pnorm' else rng.uniform(0.4, 0.85) if name == 'power' else 2.0)
    s['drive_db'] = rng.uniform(-6, 30) if name != 'power' else rng.uniform(10, 70)      # the loudest value of the sound against the curve's knee (power: against where it stops being a line)
    s['bias'] = rng.uniform(-0.6, 0.6) if rng.random() < 0.3 else 0.0                   # an offset in front of the curve: the two halves bent unevenly
    s['skew'] = LU(0.5, 2.0) if rng.random() < 0.15 else 1.0                            # one half driven harder than the other
    s['dry'] = rng.uniform(0.1, 0.7) if rng.random() < 0.2 else 0.0                     # plain sound mixed in (at the level the driven sound has before the curve bends it)
    s['block'] = LU(5, 40) if rng.random() < 0.5 else 0.0                               # an offset blocker behind the curve (Hz)
    s['muffle'] = LU(3000, 16000) if rng.random() < 0.3 else 0.0                        # a mild low-pass behind the curve (Hz)
    s['muffle_n'] = int(rng.choice([1, 2])); s['pre'] = rng.uniform(-3, 3) if rng.random() < 0.25 else 0.0        # a tilt in front (dB per octave around 1 kHz)
    return s
def run(x, s):
    """x (n, ch) -> (y, u): the box's output and the sound as it reaches the curve (scaled so that the curve's knee is at 1)."""
    x = np.asarray(x, float)
    if s['pre']:
        n = len(x); f = np.fft.rfftfreq(n, 1 / SR); g = 10 ** (s['pre'] * np.log2(np.maximum(f, 20) / 1000.0) / 20); x = np.fft.irfft(np.fft.rfft(x, axis=0) * g[:, None], n, axis=0)
    pk = np.abs(x).max() + 1e-12; u = x / pk * 10 ** (s['drive_db'] / 20); f = curve(s['name'], s['p']); v = u.copy()
    if s['skew'] != 1.0: v = np.where(v > 0, v * s['skew'], v)
    y = f(v + s['bias']) - f(np.array(s['bias']))
    if s['dry']: y = (1 - s['dry']) * y + s['dry'] * u / max(10 ** (s['drive_db'] / 20), 1.0)
    if s['block']: b, a = ss.butter(1, s['block'] / (SR / 2), 'high'); y = ss.lfilter(b, a, y, axis=0)
    if s['muffle']: b, a = ss.butter(s['muffle_n'], s['muffle'] / (SR / 2), 'low'); y = ss.lfilter(b, a, y, axis=0)
    return y, u
def limiter(x, rng):
    """Not a bend: a level held down by a moving gain (a limiter or a hard compressor). The waveform keeps its shape."""
    thr = 10 ** (rng.uniform(-18, -3) / 20) * np.abs(x).max(); env = np.abs(x).max(1); w = int(SR * 10 ** rng.uniform(-3.3, -2.0))
    pk = np.maximum.reduce([np.roll(env, k) for k in range(0, w, max(1, w // 16))]); a = np.exp(-1 / (SR * 10 ** rng.uniform(-2.0, -0.7))); rel = np.empty_like(pk); acc = 0.0
    rel = ss.lfilter([1 - a], [1, -a], pk); held = np.maximum(pk, rel); g = np.minimum(1.0, thr / np.maximum(held, 1e-9)); return x * g[:, None]
