# Hand-made clues per band and moment for the quick model: the level against its own past and future at many time scales,
# against the neighbouring bands, and what the stereo picture says. All from LE, U64, U256 (T, 41) in dB.
import numpy as np, scipy.signal as ss, scipy.ndimage as ndi
DT = 512 / 48000.0
def shift(a, k, fill):
    """a[t+k] with fill outside."""
    out = np.full_like(a, fill)
    if k > 0: out[:-k] = a[k:]
    elif k < 0: out[-k:] = a[:k]
    else: out[:] = a
    return out
def leaky(P, tau, back=False):
    """One-pole average of power over the past (not counting the present frame); back=True: over the future."""
    a = np.exp(-DT / tau); x = P[::-1] if back else P; y = ss.lfilter([0.0, 1.0 - a], [1.0, -a], x, axis=0); return y[::-1] if back else y
NAMES = []
def features(LE, U64, U256):
    T, B = LE.shape; top = LE.max(); L = np.maximum(LE - top, -80.0); P = 10 ** (L / 10); F = []; N = []
    def add(n, v): N.append(n); F.append(v.astype(np.float32))
    add('level', L)
    for k in (1, 2, 4, 8, 16, 32, 64, 128): add('past%d' % k, shift(L, -k, -80.0) - L)
    for k in (1, 2, 4, 8, 16, 32): add('next%d' % k, shift(L, k, -80.0) - L)
    for tau in (0.03, 0.1, 0.3, 1.0, 3.0): add('pastavg%g' % tau, 10 * np.log10(leaky(P, tau) + 1e-8) - L)
    for tau in (0.03, 0.1, 0.3): add('nextavg%g' % tau, 10 * np.log10(leaky(P, tau, True) + 1e-8) - L)
    for sec in (0.25, 1.0, 3.0):
        k = int(sec / DT); add('pastmax%g' % sec, ndi.maximum_filter1d(L, k, axis=0, origin=(k - 1) // 2, mode='constant', cval=-80.0) - L)
    k = int(0.3 / DT); add('nextmax0.3', ndi.maximum_filter1d(L, k, axis=0, origin=-((k - 1) // 2), mode='constant', cval=-80.0) - L)
    k = int(0.5 / DT); add('pastmin0.5', ndi.minimum_filter1d(L, k, axis=0, origin=(k - 1) // 2, mode='constant', cval=0.0) - L)
    for j in (1, 2, 4):
        up = np.concatenate([L[:, j:], np.repeat(L[:, -1:], j, 1)], 1); dn = np.concatenate([np.repeat(L[:, :1], j, 1), L[:, :-j]], 1); add('up%d' % j, up - L); add('down%d' % j, dn - L)
    Lb = 10 * np.log10(P.mean(1, keepdims=True) + 1e-8) * np.ones((1, B)); add('allbands', Lb - L); add('allbands_past8', shift(Lb, -8, -80.0) - Lb); add('allbands_next8', shift(Lb, 8, -80.0) - Lb)
    u1 = np.clip(U64, -40.0, 0.0); u2 = np.clip(U256, -40.0, 0.0); add('unshared64', u1); add('unshared256', u2)
    for k in (-8, -3, 3, 8): add('unshared64_%+d' % k, shift(u1, k, 0.0))
    k = int(0.5 / DT); add('unshared_pastmin', ndi.minimum_filter1d(u1, k, axis=0, origin=(k - 1) // 2, mode='constant', cval=0.0)); add('unshared_nextmax', ndi.maximum_filter1d(u1, k, axis=0, origin=-((k - 1) // 2), mode='constant', cval=-40.0))
    live = L > -50.0
    for q in (10, 50, 90):
        v = np.array([np.percentile(u2[live[:, b], b], q) if live[:, b].sum() > 20 else -20.0 for b in range(B)]); add('unshared_file%d' % q, np.broadcast_to(v[None, :], (T, B)))
    for q in (50, 90):
        v = np.array([np.percentile(L[:, b], q) for b in range(B)]); add('level_file%d' % q, np.broadcast_to(v[None, :], (T, B)) - L)
    add('band', np.broadcast_to(np.arange(B, dtype=np.float32)[None, :], (T, B)))
    NAMES[:] = N; return np.stack(F, -1)            # (T, B, features)
def target(LE, LS):
    top = LE.max(); fl = top - 60.0; live = LE > fl; return np.clip(np.maximum(LS, fl) - LE, -30.0, 0.0), live
