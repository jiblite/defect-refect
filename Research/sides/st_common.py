# Common pieces for the study of the stereo effects (HAAS, WIDTH): the cases, a score that hears the stereo picture too.
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, scipy.signal as ss, soundfile as sf
import truth, score
SR = 48000; SETS = '/tmp/claude-0/v2/bench/sets/'
def cases(fam): return sorted(glob.glob(SETS + fam + '/*/'))
def ms(x): return 0.5 * (x[:, 0] + x[:, 1]), 0.5 * (x[:, 0] - x[:, 1])
def lr(m, s): return np.stack([m + s, m - s], 1)
def db(a, b=1.0): return 10 * np.log10(max(float(np.sum(np.square(a))), 1e-30) / max(float(np.sum(np.square(b))) if not np.isscalar(b) else b, 1e-30))
def pictures4(x):
    """Log-mel power pictures of left, right, middle and side at the three resolutions of score.py."""
    m, s = ms(x); out = []
    for n_fft, hop, n_mels in score.RES:
        P = []
        for v in (x[:, 0], x[:, 1], m, s):
            f, t, Z = ss.stft(v, SR, 'hann', n_fft, n_fft - hop, boundary=None, padded=False); P.append(score._mel(n_fft, n_mels) @ (np.abs(Z) ** 2))
        out.append(np.stack(P))
    return out
def distance4(a, b, floor_db=-60.0, cap_db=30.0):
    """score.distance with the middle and the side as two more pictures: a sound whose two sides are right but whose
    stereo picture is wrong (one side late, a side signal that should not be there) is far away."""
    n = min(len(a), len(b)); A = pictures4(a[:n]); B = pictures4(b[:n]); d = []
    for PA, PB in zip(A, B):
        top = PA.max() + 1e-20; LA = 10 * np.log10(np.maximum(PA, top * 10 ** (floor_db / 10))); LB0 = 10 * np.log10(np.maximum(PB, 1e-30))
        live = (LA[:2] > LA.min() + 6); g = np.median((LA[:2] - LB0[:2])[live]) if live.any() else 0.0       # level match on left and right only
        LB = np.maximum(LB0 + g, LA.min()); d.append(np.minimum(np.abs(LA - LB), cap_db).mean())
    return float(np.mean(d))
def removed4(dry, wet, out):
    dw = distance4(dry, wet); return 100.0 * (1.0 - distance4(dry, out) / max(dw, 1e-9)), dw
