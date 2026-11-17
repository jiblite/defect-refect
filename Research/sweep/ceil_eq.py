# A ceiling for "a correction per moment of the spectrum": the TRUE moving magnitude response of each case (wet against dry, per
# frame and frequency, smoothed a little) used to take the effect out of the wet sound (divide) and to put it on the other
# sound (multiply). No phase, no model: what a perfect blind measurement of the moving tone curve could reach at best.
#   python3 ceil_eq.py family [smooth: bins-per-octave divisor, default 16]
import sys, numpy as np, scipy.signal as ss, scipy.ndimage as ndi
sys.path.insert(0, '/tmp/claude-0/v2/sweep'); from lab import *
fam = sys.argv[1]; DIV = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0; N, H = 2048, 256
def stft(x): return np.stack([ss.stft(x[:, c], SR, 'hann', N, N - H, boundary='zeros', padded=True)[2] for c in range(2)])          # (2, bins, frames)
def istft(Z, n): return np.stack([ss.istft(Z[c], SR, 'hann', N, N - H, boundary=True)[1][:n] for c in range(2)], 1)
def smooth(P):
    """Power picture smoothed over 3 frames and over a width that grows with frequency (a 1/DIV of the bin number, at least 3 bins)."""
    P = ndi.uniform_filter1d(P, 3, axis=-1, mode='nearest'); cs = np.concatenate([np.zeros(P.shape[:1] + (1,) + P.shape[2:]), np.cumsum(P, 1)], 1); nb = P.shape[1]; out = np.empty_like(P)
    for k in range(nb):
        h = max(1, int(round(k / DIV / 2))); a = max(0, k - h); b = min(nb, k + h + 1); out[:, k] = (cs[:, b] - cs[:, a]) / (b - a)
    return out
def response(dry, wet):
    X = stft(dry); Y = stft(wet); Px = smooth(np.abs(X) ** 2); Py = smooth(np.abs(Y) ** 2); top = Px.max(); ok = Px > top * 1e-7
    G = np.where(ok, Py / np.maximum(Px, 1e-30), np.nan)
    for c in range(2):                                   # where the dry sound has nothing: the frame's typical value
        med = np.nanmedian(G[c], 0); med = np.where(np.isfinite(med), med, 1.0); G[c] = np.where(np.isfinite(G[c]), G[c], med[None, :])
    return np.clip(G, 1e-4, 1e2)
rows = []
for d in cases(fam):
    cA = load(d, 'A'); cB = load(d, 'B')
    for c, o, tag in ((cA, cB, 'A'), (cB, cA, 'B')):
        G = response(c['dry'], c['wet']); n = len(c['wet'])
        out = istft(stft(c['wet']) / np.sqrt(np.maximum(G, 1e-3)), n); onto = istft(stft(o['dry']) * np.sqrt(G), len(o['dry']))
        r = score.removed(c['dry'], c['wet'], out)[0]; ro = score.removed(o['wet'], o['dry'], onto)[0]; rows.append((r, ro))
        print('%-20s %s out %7.1f onto %7.1f' % (c['name'], tag, r, ro), flush=True)
v = np.array(rows); print('%s: moving-tone-curve ceiling out: median %.1f mean %.1f worse %d/%d | onto: median %.1f mean %.1f worse %d/%d' % (fam, np.median(v[:, 0]), v[:, 0].mean(), (v[:, 0] < 0).sum(), len(v), np.median(v[:, 1]), v[:, 1].mean(), (v[:, 1] < 0).sum(), len(v)))
