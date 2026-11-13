# De-reverb and reverb boost, prototype: per frequency bin, the late-reverb power is tracked from the sound's own history with
# the measured room (tau, A per band), and the bins are turned down (or up) by how much of them is room.
import sys
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, scipy.signal as ss, numba, room
SR = 48000; NF, HOP = 2048, 512; DT = HOP / SR
_win = np.sqrt(ss.get_window('hann', NF, fftbins=True))
def fwd(x):
    """(n,2) -> complex frames (2, T, bins); a window of silence in front and behind."""
    xp = np.vstack([np.zeros((NF, 2)), x, np.zeros((NF + HOP, 2))]); n = 1 + (len(xp) - NF) // HOP
    idx = np.arange(NF)[None, :] + HOP * np.arange(n)[:, None]; return np.stack([np.fft.rfft(xp[idx, c] * _win, axis=1) for c in range(2)])
def inv(X, n):
    T = X.shape[1]; out = np.zeros((NF + HOP * (T - 1), 2)); norm = np.zeros(NF + HOP * (T - 1))
    for c in range(2):
        fr = np.fft.irfft(X[c], NF, axis=1) * _win
        for t in range(T): out[t * HOP:t * HOP + NF, c] += fr[t]
    for t in range(T): norm[t * HOP:t * HOP + NF] += _win ** 2
    return (out / np.maximum(norm, 1e-9)[:, None])[NF:NF + n]
@numba.njit(cache=True)
def _track_bins(E, a, g):
    T, F = E.shape; R = np.zeros((T, F)); r = np.zeros(F)
    for t in range(T):
        for f in range(F):
            R[t, f] = r[f]; s = E[t, f] - r[f]
            if s < 0.0: s = 0.0
            r[f] = a[f] * r[f] + g[f] * s
    return R
def to_bins(v, fc=room.FC):
    """Band values -> per-bin values, along log frequency."""
    f = np.fft.rfftfreq(NF, 1 / SR); return np.interp(np.log(np.maximum(f, 20.0)), np.log(fc), v)
def smooth_tf(P, kt=3, kf=3):
    return ss.fftconvolve(ss.fftconvolve(P, np.ones((kt, 1)) / kt, 'same'), np.ones((1, kf)) / kf, 'same')
def reverb_power(E, tau_b, A_b):
    """Late-reverb power per bin and frame predicted from the power picture E (T, bins) and the room per band."""
    tau = np.maximum(to_bins(tau_b), 0.01); A = np.exp(to_bins(np.log(np.maximum(A_b, 1e-6))))
    return _track_bins(np.ascontiguousarray(E), np.exp(-DT / tau), A * DT)
def gain_from(E, R, s, floor_db=-30.0, over=1.0):
    """Gain per bin for amount s: 0 = reverb out, 1 = untouched, 2 = reverb twice as loud (amplitude)."""
    frac = np.clip(over * R / np.maximum(E, 1e-30), 0.0, 1.0)
    if s <= 1.0:
        g_full = np.maximum(np.sqrt(1.0 - frac), 10 ** (floor_db / 20)); return g_full ** (1.0 - s)
    return np.sqrt(1.0 + (s * s - 1.0) * frac)
def process(x, tau_b, A_b, s=0.0, floor_db=-30.0, over=1.0, R=None, ret=False):
    X = fwd(x); E = np.abs(X[0]) ** 2 + np.abs(X[1]) ** 2; Es = smooth_tf(E)
    Rh = reverb_power(Es, tau_b, A_b) if R is None else R
    G = gain_from(Es, Rh, s, floor_db, over); y = inv(X * G[None], len(x))
    return (y, G, Rh, Es) if ret else y
