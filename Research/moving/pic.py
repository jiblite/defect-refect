# The picture the moving-tone engine reads (band levels over time, 12 bands an octave), and how a moving tone curve given on
# that picture is put on a sound or taken off it again.
import numpy as np, scipy.fft as sfft
SR = 48000; N = 2048; HOP = 1024; BPO = 12; F0 = 70.0; NB = 96
FC = F0 * 2.0 ** (np.arange(NB) / BPO)                       # band centres, 70 Hz .. 16.9 kHz
RN = 2048; RH = 512                                          # the rendering transform: four windows over every sample
FLOOR = 70.0                                                 # a cell this far (dB) under the loudest cell counts as silent

def _weights():
    fb = np.arange(N // 2 + 1) * SR / N; W = np.zeros((NB, N // 2 + 1))
    for b in range(NB):
        lo = min(FC[b] * 2.0 ** (-1.0 / BPO), FC[b] - SR / N); hi = max(FC[b] * 2.0 ** (1.0 / BPO), FC[b] + SR / N)
        W[b] = np.clip(np.minimum((fb - lo) / (FC[b] - lo), (hi - fb) / (hi - FC[b])), 0.0, 1.0); W[b] /= W[b].sum()
    return W
W = _weights(); WIN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)

def frames(x, n=N, hop=HOP):
    """Frames centred on the samples 0, hop, 2 hop ...: (T, n), T = len(x) // hop + 1."""
    T = len(x) // hop + 1; xp = np.concatenate([np.zeros(n // 2), x, np.zeros(n + hop)])
    return xp[np.arange(T)[:, None] * hop + np.arange(n)[None, :]]

def power(x):
    """Band power per frame (T, NB) of one channel."""
    return (np.abs(sfft.rfft(frames(x) * WIN, axis=1)) ** 2) @ W.T

def picture(x):
    """Band level in dB per frame (T, NB)."""
    return 10.0 * np.log10(power(x) + 1e-13)

def inputs(LE):
    """Level picture (T, NB) in dB -> what the model reads (4, T, NB): the level under the loudest cell, the level against the
    band's own middle level over the sound, that middle level, and the band's place."""
    top = LE.max(); live = LE > top - FLOOR; e = np.clip((LE - top) / 20.0, -3.5, 0.0) + 1.75
    med = np.full(LE.shape[1], top - FLOOR)
    for b in range(LE.shape[1]):
        if live[:, b].any(): med[b] = np.median(LE[live[:, b], b])
    dev = np.where(live, np.clip(LE - med[None, :], -30.0, 30.0) / 15.0, -2.0)
    mp = np.broadcast_to((np.clip((med - top) / 20.0, -3.5, 0.0) + 1.75)[None, :], LE.shape); pos = np.broadcast_to(np.linspace(-1.0, 1.0, LE.shape[1])[None, :], LE.shape)
    return np.stack([e, dev, mp, pos]).astype(np.float32)

_fb = np.arange(RN // 2 + 1) * SR / RN; _lb = np.log2(np.maximum(_fb, 1.0) / F0) * BPO      # every bin's place on the band axis
_i0 = np.clip(np.floor(_lb).astype(int), 0, NB - 2); _fr = np.clip(_lb - _i0, 0.0, 1.0)
_rwin = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(RN) / RN))

def render(x, G, sign=1.0, lift=24.0, cut=60.0):
    """One channel x with the tone curve G (T, NB; dB per frame of the picture and band) put on (sign 1) or taken off (-1),
    never lifting by more than `lift` dB nor cutting by more than `cut`."""
    n = len(x); T = G.shape[0]; J = n // RH + 1 + RN // RH; xp = np.concatenate([np.zeros(RN), x, np.zeros(RN + RH)]); out = np.zeros(len(xp)); nrm = np.zeros(len(xp))
    for j in range(J):
        s = j * RH; t = (s - RN // 2) / HOP; t = min(max(t, 0.0), T - 1.0); k = min(int(t), T - 2) if T > 1 else 0; f = t - k          # the frame's centre on the picture's time axis
        g = (G[k] * (1 - f) + G[min(k + 1, T - 1)] * f) if T > 1 else G[0]
        gb = np.clip(sign * (g[_i0] * (1 - _fr) + g[_i0 + 1] * _fr), -cut, lift)
        Z = sfft.rfft(xp[s:s + RN] * _rwin) * 10.0 ** (gb / 20.0); out[s:s + RN] += sfft.irfft(Z, RN) * _rwin; nrm[s:s + RN] += _rwin ** 2
    return (out / np.maximum(nrm, 1e-9))[RN:RN + n]

def probe(n, seed=0, level=0.05):
    """A steady test signal that repeats every N samples (every frame of the picture sees the same spectrum), with a little
    less in the highs: what goes through an effect to read its moving tone curve off the picture."""
    r = np.random.default_rng(seed); k = np.arange(1, N // 2); ph = r.uniform(0, 2 * np.pi, len(k)); a = k ** -0.25
    spec = np.zeros(N // 2 + 1, complex); spec[1:N // 2] = a * np.exp(1j * ph); p = sfft.irfft(spec, N); p *= level / np.sqrt((p ** 2).mean())
    return np.tile(p, n // N + 2)[:n]

def curve(dry_probe, wet_probe, lo=-40.0, hi=20.0):
    """The tone curve an effect put on the probe: (T, NB) dB."""
    return np.clip(10.0 * np.log10((power(wet_probe) + 1e-16) / (power(dry_probe) + 1e-16)), lo, hi)
