# Tape-speed change along a sound: the sound read at a moving position with a band-limited (windowed sinc) reader.
# This is the PITCH fader's whole way of changing a sound: nothing is stretched or re-synthesised, so there is no
# warble and nothing gets dull; what moves is the timing inside the file.
import numpy as np

H = 16          # zero crossings of the reader on each side (at the cutoff's own scale)
PH = 512        # table steps per zero crossing
_tab = None
def _table():
    global _tab
    if _tab is None:
        u = np.arange(-H * PH, H * PH + 1) / PH
        _tab = np.sinc(u) * (0.5 + 0.5 * np.cos(np.pi * u / H))       # Hann-windowed sinc
    return _tab

def read_at(x, pos, chunk=8192):
    """x (n,) or (n,c) read at the positions pos (in samples, rising). Where the positions move faster than one
    sample per sample (pitch up) the reader's cutoff comes down with them, so nothing folds back."""
    tab = _table(); one = x.ndim == 1; X = x[:, None] if one else x; n = len(X)
    pos = np.asarray(pos, float); step = np.gradient(pos) if len(pos) > 1 else np.ones(1)
    fc = np.minimum(1.0, 1.0 / np.maximum(step, 1e-6)); out = np.zeros((len(pos), X.shape[1]))
    for a in range(0, len(pos), chunk):
        p = pos[a:a + chunk]; f = fc[a:a + chunk]; T = int(np.ceil(H / f.min())) + 1
        k = np.floor(p).astype(np.int64)[:, None] + np.arange(-T + 1, T + 1)[None, :]          # taps
        u = (k - p[:, None]) * f[:, None]; ui = np.clip(u, -H, H) * PH + H * PH
        i0 = np.minimum(ui.astype(np.int64), 2 * H * PH - 1); fr = ui - i0
        w = (tab[i0] * (1 - fr) + tab[i0 + 1] * fr) * f[:, None]; w[np.abs(u) >= H] = 0.0
        ok = (k >= 0) & (k < n); kk = np.clip(k, 0, n - 1)
        for c in range(X.shape[1]): out[a:a + chunk, c] = (X[kk, c] * w * ok).sum(1)
    return out[:, 0] if one else out

def positions(cents, n_in=None, keep_length=True):
    """Where to read for a pitch change of cents[t] (per output sample): the running sum of the speed 2^(cents/1200).
    keep_length: the speed is scaled as a whole so that the reader ends exactly at the sound's end (len(cents) out for n_in in)."""
    r = 2.0 ** (np.asarray(cents, float) / 1200.0); pos = np.concatenate([[0.0], np.cumsum(r)[:-1]])
    if keep_length:
        n_in = n_in or len(r); k = (n_in - 1) / max(pos[-1], 1e-9); pos = pos * k
    return pos

def bend(x, cents, keep_length=False):
    """x with its pitch moved by cents[t] at output sample t (same number of samples out)."""
    pos = positions(cents, len(x), keep_length); return read_at(x, pos), pos
