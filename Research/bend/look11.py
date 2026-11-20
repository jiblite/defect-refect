# What the meter makes of made steady tones (no bend in any of them): sines, harmonic sounds, chords, plain synthesizer waves.
import sys
from common import *
import bfeat, bengine, bmeter
M = bmeter.model(sys.argv[1]); rng = np.random.default_rng(5); SR = 48000; n = 5 * SR; t = np.arange(n) / SR; out = {}
def harm(f0, amps, phases, vib=0.0, trem=0.0):
    ph = 2 * np.pi * f0 * t + (vib * f0 / 5.0) * np.sin(2 * np.pi * 5.0 * t); y = sum(a * np.sin(k * ph + p) for k, (a, p) in enumerate(zip(amps, phases), 1) if k * f0 < 20000)
    return y * (1 + trem * np.sin(2 * np.pi * rng.uniform(0.3, 6) * t))
def read(name, y, noise=-60.0):
    y = y / np.abs(y).max() * 0.5; y = y + rng.standard_normal(n) * 0.5 * 10 ** (noise / 20); y = np.stack([y, y], 1); X, vp = bfeat.features(y); S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1]))
    top = max(bengine.tidy(S[0], bengine.SPREAD)[-1], bengine.tidy(S[1], bengine.SPREAD)[-1]); out.setdefault(name, []).append((logit, top))
for i in range(12):
    f0 = float(np.exp(rng.uniform(np.log(60), np.log(2000)))); K = int(rng.integers(2, 30)); ph = rng.uniform(0, 2 * np.pi, 64); z = np.zeros(64)
    read('sine', harm(f0, [1.0], [0.0])); read('sine, wavering', harm(f0, [1.0], [0.0], rng.uniform(0, 0.02), rng.uniform(0.05, 0.5)))
    read('harmonics falling off, any phases', harm(f0, [1.0 / k ** rng.uniform(0.5, 2.5) for k in range(1, K + 1)], ph, rng.uniform(0, 0.01), rng.uniform(0, 0.3)))
    read('harmonics of any strength', harm(f0, list(rng.uniform(0, 1, K) / np.arange(1, K + 1) ** 0.5), ph, rng.uniform(0, 0.01), rng.uniform(0, 0.3)))
    read('sawtooth', harm(f0, [1.0 / k for k in range(1, 400)], np.zeros(400))); read('square', harm(f0, [(k % 2) / k for k in range(1, 400)], np.zeros(400)))
    read('triangle', harm(f0, [(k % 2) * (-1) ** (k // 2) / k ** 2 for k in range(1, 400)], np.zeros(400)))
    read('pulse (a fifth of the time up)', harm(f0, [np.sin(np.pi * k * 0.2) / k for k in range(1, 400)], np.full(400, np.pi / 2)))
    read('chord of three sines', sum(harm(f0 * r, [1.0], [rng.uniform(0, 6.28)]) for r in (1.0, 1.26, 1.5)))
    read('two close sines (beating)', harm(f0, [1.0], [0.0]) + harm(f0 * (1 + rng.uniform(0.002, 0.02)), [1.0], [1.0]))
    read('sine in noise at -20 dB', harm(f0, [1.0], [0.0]), -20.0); read('noise alone', rng.standard_normal(n) * 1e-3, 0.0)
for k, v in out.items():
    v = np.array(v); print('%-36s sure: median %5.1f, most %5.1f | read at the top: median %4.1f dB, most %4.1f | shown at the default on %2d of %d, at the lowest on %2d' % (k, np.median(v[:, 0]), v[:, 0].max(), np.median(v[:, 1]), v[:, 1].max(),
          ((v[:, 0] >= bengine.SURE_AT) & (v[:, 1] >= bengine.SURE_TOP)).sum(), len(v), ((v[:, 0] >= bengine.LOW_AT) & (v[:, 1] >= bengine.TOP_AT)).sum()))
