# From band gains to sound: the gains (dB, per band and frame) are spread over the FFT bins along log frequency and applied.
import sys
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, common as cm
_lf = np.log(np.maximum(np.fft.rfftfreq(cm.NF, 1 / cm.SR), 20.0)); _lc = np.log(cm.FC41)
def to_bins(v):
    """(T, 41) -> (T, bins), straight lines between band centres along log frequency."""
    i = np.clip(np.searchsorted(_lc, _lf) - 1, 0, cm.NB - 2); w = np.clip((_lf - _lc[i]) / (_lc[i + 1] - _lc[i]), 0.0, 1.0); return v[:, i] * (1 - w)[None, :] + v[:, i + 1] * w[None, :]
def gains(g_db, s=0.0):
    """Bin gains for amount s: 0 = reverb out (the model's full answer), 1 = untouched, 2 = the reverb part twice as loud."""
    gb = to_bins(g_db)
    if s <= 1.0: return 10 ** (gb * (1.0 - s) / 20.0)
    rho = 1.0 - 10 ** (gb / 10.0); return np.sqrt(1.0 + (s * s - 1.0) * rho)
def render(wet, g_db, s=0.0):
    X = cm.fwd(wet); return cm.inv(X * gains(g_db, s)[None], len(wet))

def reshaped(wet, g_db, s, F, width=1.0):
    """As render(), with the room part of every band and moment made F times as strong (F: (T, 41), power; 1 = as it is)
    and its width changed (width 0..2: the room part of the side signal is scaled, 1 = as it is). Per bin the power is
    split by the gains into a direct share G and a room share 1 - G; the result holds G + A * F * (1 - G) of it, where A
    is what the Amount fader alone leaves of the room share (so with F = 1 and width = 1 this is render() exactly)."""
    X = cm.fwd(wet); G = 10 ** (to_bins(g_db) / 10.0); Fb = 10 ** (to_bins(10 * np.log10(np.maximum(F, 1e-6))) / 10.0); R = 1.0 - G
    A = np.where(R > 1e-9, (G ** (1.0 - s) - G) / np.maximum(R, 1e-9), s) if s <= 1.0 else s * s
    gm = np.sqrt(G + A * Fb * R)
    if abs(width - 1.0) < 1e-9: return cm.inv(X * gm[None], len(wet))
    gs = np.sqrt(G + A * Fb * R * width * width); M = 0.5 * (X[0] + X[1]); S = 0.5 * (X[0] - X[1]); return cm.inv(np.stack([M * gm + S * gs, M * gm - S * gs]), len(wet))
