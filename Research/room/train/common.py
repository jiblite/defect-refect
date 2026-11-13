# Shared by training-data making, training and (later) the app port: the band layout and what is computed from a sound.
import numpy as np, scipy.signal as ss
SR = 48000; NF, HOP = 2048, 512; DT = HOP / SR
_win = np.sqrt(ss.get_window('hann', NF, fftbins=True))
def edges41():
    """Band edges in FFT bins: 7 bands of two bins (47..375 Hz), then 34 bands a sixth of an octave wide, up to 19 kHz."""
    e = list(range(2, 17, 2)); f = 375.0
    for k in range(1, 35): e.append(int(round(375.0 * 2 ** (k / 6.0) / (SR / NF))))
    return np.array(e)
EDGES = edges41(); NB = len(EDGES) - 1; FC41 = np.sqrt(EDGES[:-1] * EDGES[1:]) * SR / NF
BM = np.zeros((NB, NF // 2 + 1))
for _b in range(NB): BM[_b, EDGES[_b]:EDGES[_b + 1]] = 1.0
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
def bandpow(X): return (np.abs(X[0]) ** 2 + np.abs(X[1]) ** 2) @ BM.T
def _box(a, k, axis):
    """Mean over k neighbours along an axis, zeros outside, centred as scipy's 'same' does it (an even k has its extra one
    on the low side). Plain sums on purpose: silence stays exactly silent. (An FFT convolution leaves rounding noise of
    about 1e-16 of the loudest value, and the unshared share of a silent cell then is the ratio of two noises.)"""
    n = a.shape[axis]; pad = [(0, 0), (0, 0)]; pad[axis] = (k // 2, (k - 1) // 2); p = np.pad(a, pad); out = np.zeros_like(a)
    for j in range(k): out += p[j:j + n] if axis == 0 else p[:, j:j + n]
    return out / k
def unshared_db(X, tavg, favg=2):
    """Per band and frame: how much of the power the two channels do not share, in dB (0 = nothing shared, -40 = one source
    at a fixed place). Per bin, the left/right covariance is averaged over tavg seconds and +-favg bins; twice its smaller
    eigenvalue is the unshared power."""
    pl = np.abs(X[0]) ** 2; pr = np.abs(X[1]) ** 2; c = X[0] * np.conj(X[1]); k = max(1, int(round(tavg / DT)))
    sm = lambda a: _box(_box(a, k, 0), 2 * favg + 1, 1)
    pls, prs, cs = sm(pl), sm(pr), sm(c.real) + 1j * sm(c.imag)
    lam2 = np.maximum((pls + prs) - np.sqrt(np.maximum((pls - prs) ** 2 + 4 * np.abs(cs) ** 2, 0.0)), 0.0)
    return 10 * np.log10((lam2 @ BM.T) / np.maximum((pls + prs) @ BM.T, 1e-30) + 1e-4)
def features(wet, direct=None):
    """What the model sees of a sound (LE: band level in dB; U64, U256: unshared share in dB at two averaging times), and,
    when the true direct part is given, what it has to find (LS, LR: level of the direct part and of the reverb part)."""
    X = fwd(wet); out = dict(LE=10 * np.log10(bandpow(X) + 1e-12), U64=unshared_db(X, 0.064), U256=unshared_db(X, 0.256))
    if direct is not None:
        out['LS'] = 10 * np.log10(bandpow(fwd(direct)) + 1e-12); out['LR'] = 10 * np.log10(bandpow(fwd(wet - direct)) + 1e-12)
    return out
