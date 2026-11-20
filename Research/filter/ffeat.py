# The fixed-filter study: what is read off a sound. Band energies per frame (1/12-octave bands, 50 Hz .. 19 kHz, both sides
# together), and from them the four curves the reader sees: per band the level it has on average and at its 10th, 50th and
# 90th percentile over the frames that count (dB against the sound's own level).
import numpy as np, scipy.signal as ss
SR = 48000; N = 8192; HOP = 2048; NBAND = 104; F0 = 50.0; FLOOR = -110.0; MAXFRAMES = 280
FC = F0 * 2.0 ** (np.arange(NBAND) / 12.0)
def band_matrix():
    f = np.fft.rfftfreq(N, 1 / SR); W = np.zeros((NBAND, len(f))); binhz = SR / N
    for i, fc in enumerate(FC):
        half = max(fc * (2 ** (1 / 12.0) - 1), 1.5 * binhz); W[i] = np.maximum(0.0, 1.0 - np.abs(f - fc) / half)
    return W
_W = band_matrix()
def energies(x):
    """x (n, 2) -> E (frames, bands): band energies, both sides together."""
    P = 0
    for c in range(x.shape[1]):
        f, t, Z = ss.stft(x[:, c], SR, 'hann', N, N - HOP, boundary=None, padded=False); P = P + np.abs(Z) ** 2
    return (_W @ P).T
def curves(E, span=50.0):
    """E (frames, bands) -> (4, bands) in dB against the sound's level, and that level; frames more than span dB under the loudest do not count."""
    tot = E.sum(1); keep = tot > tot.max() * 10 ** (-span / 10)
    if keep.sum() < 4: keep = tot >= np.sort(tot)[-min(4, len(tot))]
    Ek = E[keep]; ref = 10 * np.log10(Ek.sum(1).mean() + 1e-30); D = 10 * np.log10(Ek + 1e-30) - ref
    C = np.stack([10 * np.log10(Ek.mean(0) + 1e-30) - ref, np.percentile(D, 10, axis=0), np.percentile(D, 50, axis=0), np.percentile(D, 90, axis=0)])
    return np.maximum(C, FLOOR), ref
def curves6(E, span=50.0):
    """As curves, with the quietest and the loudest frame of every band as two more curves: (6, bands). A long sound is
    looked at in every second frame, every third ...: never more than MAXFRAMES of them."""
    E = E[::(len(E) + MAXFRAMES - 1) // MAXFRAMES]; tot = E.sum(1); keep = tot > tot.max() * 10 ** (-span / 10)
    if keep.sum() < 4: keep = tot >= np.sort(tot)[-min(4, len(tot))]
    Ek = E[keep]; ref = 10 * np.log10(Ek.sum(1).mean() + 1e-30); D = np.sort(10 * np.log10(Ek + 1e-30) - ref, 0); n = len(D)
    q = lambda p: D[int(round(p * (n - 1)))]            # the frame at this place in the order (no interpolation: the same in the app)
    C = np.stack([10 * np.log10(Ek.mean(0) + 1e-30) - ref, D[0], q(0.1), q(0.5), q(0.9), D[-1]])
    return np.maximum(C, FLOOR), ref
