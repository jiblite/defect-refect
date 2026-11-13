# Scores: how far is sound b from sound a, on a scale that follows what is heard (log-mel pictures at three time scales).
import numpy as np, scipy.signal as ss, librosa
SR = 48000
_fb = {}
def _mel(n_fft, n_mels):
    k = (n_fft, n_mels)
    if k not in _fb: _fb[k] = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=30, fmax=20000, norm=None)
    return _fb[k]
RES = [(512, 128, 40), (2048, 512, 64), (8192, 2048, 80)]
def pictures(x):
    """x: (n,2) or (n,). Returns log-mel power pictures (dB) for both channels at the three resolutions."""
    if x.ndim == 1: x = np.stack([x, x], 1)
    out = []
    for n_fft, hop, n_mels in RES:
        P = []
        for c in range(2):
            f, t, Z = ss.stft(x[:, c], SR, 'hann', n_fft, n_fft - hop, boundary=None, padded=False)
            P.append(_mel(n_fft, n_mels) @ (np.abs(Z) ** 2))
        out.append(np.stack(P))
    return out
def distance(a, b, floor_db=-60.0, cap_db=30.0):
    """Mean absolute dB difference between the pictures of a (the reference) and b, after the best overall level match.
    Cells quieter than floor_db under the reference's loudest cell count as the floor (silence compares equal)."""
    n = min(len(a), len(b)); A = pictures(a[:n]); B = pictures(b[:n]); d = []
    for PA, PB in zip(A, B):
        top = PA.max() + 1e-20; LA = 10 * np.log10(np.maximum(PA, top * 10 ** (floor_db / 10))); LB0 = 10 * np.log10(np.maximum(PB, 1e-30))
        live = (LA > LA.min() + 6); g = np.median((LA - LB0)[live]) if live.any() else 0.0           # the level match: median difference where the reference is audible
        LB = np.maximum(LB0 + g, LA.min())
        d.append(np.minimum(np.abs(LA - LB), cap_db).mean())
    return float(np.mean(d))
def removed(dry, wet, out):
    """How much of the effect is gone: 100 = the result matches the dry sound, 0 = no better than the wet sound, below 0 = worse."""
    dw = distance(dry, wet); return 100.0 * (1.0 - distance(dry, out) / max(dw, 1e-9)), dw
class Scorer:
    """distance() against one fixed reference, with the reference's pictures computed once."""
    def __init__(self, ref, floor_db=-60.0, cap_db=30.0):
        self.n = len(ref); self.cap = cap_db; self.ref = []
        for PA in pictures(ref):
            top = PA.max() + 1e-20; LA = 10 * np.log10(np.maximum(PA, top * 10 ** (floor_db / 10))); self.ref.append((LA, LA > LA.min() + 6))
    def __call__(self, b):
        assert len(b) >= self.n; d = []
        for (LA, live), PB in zip(self.ref, pictures(b[:self.n])):
            LB0 = 10 * np.log10(np.maximum(PB, 1e-30)); g = np.median((LA - LB0)[live]) if live.any() else 0.0
            d.append(np.minimum(np.abs(LA - np.maximum(LB0 + g, LA.min())), self.cap).mean())
        return float(np.mean(d))
