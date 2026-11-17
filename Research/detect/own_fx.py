# The app's own effects (the first prototype's Python: fxlib.py, fxlib27.py), with the settings ranges the detectors were
# trained on, for the 17 families the app shows. Used for training positives only; the scoreboard and the held-out
# test use outside effects (zoo_fx.py).
import sys
import paths; sys.path.insert(0, paths.PROTO)
import numpy as np
import fxlib as _f                      # first: fxlib27 then finds it loaded
import fxlib27 as _f27
SR = 48000
FAMILIES = ['flanger', 'chorus', 'phaser', 'haas', 'width', 'delay', 'distortion', 'bitcrush', 'filtersweep', 'tremolo', 'autopan', 'saturation', 'exciter', 'ringmod', 'filter', 'stutter', 'gatedrhythm']
STEREO_IN = ('haas', 'width', 'autopan', 'chorus', 'delay', 'flanger', 'phaser')     # these make their own stereo picture
def U(r, a, b): return float(r.uniform(a, b))
def LU(r, a, b): return float(np.exp(r.uniform(np.log(a), np.log(b))))
def sample(fam, r):
    if fam == 'flanger': return dict(rate=U(r, 0.1, 2), dmin=U(r, 0.2e-3, 2e-3), dmax=U(r, 2.5e-3, 10e-3), depth=U(r, 0.4, 0.95), fb=float(r.choice([0, 0, U(r, 0.1, 0.6)])), stereo_phase=float(r.choice([0, 0, np.pi / 2])))
    if fam == 'chorus': return dict(rate=U(r, 0.2, 2.5), base=U(r, 8e-3, 30e-3), width=U(r, 1e-3, 8e-3), voices=int(r.integers(1, 4)), mix=U(r, 0.3, 0.8))
    if fam == 'phaser': return dict(rate=U(r, 0.1, 2), stages=int(r.choice([2, 4, 6, 8, 12])), fmin=U(r, 100, 500), fmax=U(r, 1500, 8000), fb=U(r, 0, 0.7), mix=U(r, 0.4, 0.6))
    if fam == 'haas': return dict(ms=U(r, 3, 35))
    if fam == 'width': return dict(amount=U(r, 0.3, 1))
    if fam == 'delay': return dict(ms=U(r, 80, 700), fb=U(r, 0.1, 0.7), mix=U(r, 0.2, 0.6), pingpong=bool(r.random() < 0.4))
    if fam == 'distortion': return dict(drive=U(r, 2, 20), kind=str(r.choice(['tanh', 'clip'])))
    if fam == 'bitcrush':
        while True:
            p = dict(bits=int(r.integers(3, 10)), down=int(r.integers(1, 10)))
            if p['bits'] <= 8 or p['down'] >= 2: return p
    if fam == 'filtersweep': return dict(rate=U(r, 0.1, 2), fmin=U(r, 150, 800), fmax=U(r, 1500, 9000), q=U(r, 1, 8))
    if fam == 'tremolo': return dict(rate=U(r, 1.5, 12), depth=U(r, 0.3, 1))
    if fam == 'autopan': return dict(rate=U(r, 0.15, 3), depth=U(r, 0.5, 1))
    if fam == 'saturation': return dict(drive=U(r, 3, 18), tone=U(r, 0, 1), kind=str(r.choice(['tape', 'soft'])))
    if fam == 'exciter': return dict(amount=U(r, 0.15, 0.8), freq=LU(r, 1500, 8000))
    if fam == 'ringmod': return dict(carrier=LU(r, 30, 2000), mix=U(r, 0.4, 1), phase=U(r, 0, 2 * np.pi), drift=U(r, 0, 0.1) if r.random() < 0.3 else 0.0)
    if fam == 'filter':
        t = str(r.choice(['low', 'low', 'high', 'high', 'band'])); c = LU(r, 300, 8000) if t == 'low' else LU(r, 150, 3000) if t == 'high' else LU(r, 300, 4000)
        return dict(type=t, cutoff=c, resonance=LU(r, 0.5, 8), slope=int(r.choice([12, 24])))
    if fam == 'stutter': return dict(slice_ms=LU(r, 30, 250), repeats=int(r.integers(2, 9)), prob=U(r, 0.25, 0.8), seed=int(r.integers(1 << 30)))
    if fam == 'gatedrhythm': return dict(rate=LU(r, 2, 16), duty=U(r, 0.2, 0.8), depth=U(r, 0.6, 1), pattern=int(r.integers(1 << 16)) | 0x1111 if r.random() < 0.6 else 0xFFFF, phase=U(r, 0, 1))
    raise KeyError(fam)
def apply(fam, y, p, tail=0.0):
    """y: (n, 2). One side's sound goes in, the effect's stereo answer comes out; a sound whose sides differ is run side by side."""
    e = _f27.EFFECTS[fam]
    if tail: y = np.vstack([y, np.zeros((int(tail * SR), 2))])
    L, R = np.ascontiguousarray(y[:, 0]), np.ascontiguousarray(y[:, 1])
    if fam in STEREO_IN:
        if np.abs(L - R).max() < 1e-7 * (np.abs(y).max() + 1e-12): return e(L, **p)
        return np.stack([e(L, **p)[:, 0], e(R, **p)[:, 1]], 1)
    return np.stack([e(L, **p)[:, 0], e(R, **p)[:, 0]], 1)
