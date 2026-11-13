# The tone model: a small 1-D convolution stack along the 1/24-octave grid. Input: long-term statistics of the sound (see feat.py),
# output: the curve in dB per grid cell (its overall level is free).
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import feat
G = feat.G; K = len(feat.NAMES)
_t = np.arange(-48, 49); _gk = np.exp(-0.5 * (_t / 12.0) ** 2); _gk /= _gk.sum()
def smooth12(v):
    """Gaussian trend, sigma 12 cells (half an octave), ends reflected. v: (..., G)."""
    vp = np.concatenate([v[..., 48:0:-1], v, v[..., -2:-50:-1]], -1); return np.apply_along_axis(lambda a: np.convolve(a, _gk, 'valid'), -1, vp)
USE = [feat.NAMES.index(k) for k in ('floor', 'p10', 'p25', 'p50', 'p75', 'p90', 'top', 'pmean', 'flux', 'onset', 'tonal')]      # rows fed to the net besides lmean (env is left out: it cannot be simulated well in training)
SC = np.array([10, 10, 10, 10, 10, 10, 10, 10, 5, 10, 5], np.float32)       # their scales in dB
CIN = len(USE) + 3
POS = np.linspace(-1, 1, G).astype(np.float32)
def weights(F):
    """Which cells count (0..1): the level of the loud frames (lmean + top) within 60..70 dB of its largest. F: (B, K, G), lmean already wet."""
    lev = F[:, 0] + F[:, feat.NAMES.index('top')]; return np.clip((lev - (lev.max(-1, keepdims=True) - 70.0)) / 10.0, 0.0, 1.0)
def net_input(F):
    """F: (B, K, G) raw features with lmean (dB) in row 0 -> (B, CIN, G) network input: level under the loudest region, lmean's detail
    around its half-octave trend, the other statistics, and the position on the frequency axis."""
    F = np.asarray(F, np.float32); lm = F[:, 0]; tr = smooth12(lm).astype(np.float32); ref = tr.max(-1, keepdims=True)
    x = np.empty((F.shape[0], CIN, G), np.float32)
    x[:, 0] = np.clip((lm - ref) / 20.0, -4.0, 1.0); x[:, 1] = np.clip((lm - tr) / 10.0, -3.0, 3.0); x[:, 2:CIN - 1] = np.clip(F[:, USE] / SC[None, :, None], -4.0, 4.0); x[:, CIN - 1] = POS[None]
    return x
class Net(nn.Module):
    """Two outputs per cell: the curve (dB) and the model's own doubt about it (b = the expected absolute error in dB, as exp of the second output)."""
    def __init__(s, ch=64, k=5, dil=(1, 2, 4, 8, 16, 8, 4, 2, 1), cin=CIN, drop=0.0, nout=2):
        super().__init__(); s.inp = nn.Conv1d(cin, ch, 1); s.drop = nn.Dropout(drop); s.nout = nout
        s.convs = nn.ModuleList([nn.Conv1d(ch, ch, k, padding=d * (k - 1) // 2, dilation=d) for d in dil]); s.out = nn.Conv1d(ch, nout, 1)
    def forward(s, x):
        h = s.inp(x)
        for c in s.convs: h = h + s.drop(c(Fn.relu(h)))
        o = s.out(Fn.relu(h)); return (o[:, 0] * 10.0, o[:, 1].clamp(-3.0, 3.0)) if s.nout == 2 else o[:, 0] * 10.0
def nparams(m): return sum(p.numel() for p in m.parameters())
def predict(m, F, doubt=False):
    """F: (B, K, G) raw features of wet sounds -> curve dB (B, G) [and the doubt b in dB]."""
    with torch.no_grad(): o = m(torch.from_numpy(net_input(F)))
    if isinstance(o, tuple): return (o[0].numpy(), np.exp(o[1].numpy())) if doubt else o[0].numpy()
    return o.numpy()
