# Lab helpers: load cases, cepstrum pictures (blind and against the dry sound), plots. Not part of the engine.
import sys, os, json, glob
sys.path += ['/tmp/claude-0/v2/proto', '/tmp/claude-0/v2/bench']
import numpy as np, scipy.fft as sfft, scipy.ndimage as ndi, scipy.signal as ss
import truth, score
SR = 48000; SETS = '/tmp/claude-0/v2/bench/sets/'
FAMS = ('flanger', 'phaser', 'chorus', 'filtersweep')

def cases(fam): return sorted(glob.glob(SETS + fam + '/*/'))
def load(d, which='A'): return truth.load(d if d.endswith('/') else d + '/', which)

def frames(x, n, hop):
    """Frames of one channel, half a window of silence in front and behind: (T, n)."""
    xp = np.concatenate([np.zeros(n // 2), x, np.zeros(n)]); T = 1 + (len(x) + n // 2) // hop
    idx = np.arange(T)[:, None] * hop + np.arange(n)[None, :]; return xp[idx]

def logspec(x, n=2048, hop=512, nfft=None):
    """ln magnitude per frame (T, nfft/2+1) and the frame energies."""
    F = frames(x, n, hop) * np.hanning(n + 1)[:n]; S = np.abs(sfft.rfft(F, nfft or n, axis=1))
    return S, (F ** 2).sum(1)

def cepgram(S, floor=1e-3, lo_hz=150.0, hi_hz=16000.0, nfft=None, trend=None):
    """Cepstrum per frame of the magnitude picture S (T, nb): ln magnitude with each frame's own floor (floor times its top),
    band limited with a taper, the smooth trend over `trend` bins taken off first. Returns (T, nfft/2+1) with lag in samples."""
    T, nb = S.shape; n = (nb - 1) * 2; f = np.arange(nb) * SR / n
    L = np.log(np.maximum(S, floor * S.max(1, keepdims=True) + 1e-30))
    if trend: L = L - ndi.uniform_filter1d(L, trend, axis=1, mode='nearest')
    w = np.zeros(nb); m = (f >= lo_hz) & (f <= hi_hz); w[m] = np.hanning(m.sum())
    L = (L - (L * w).sum(1, keepdims=True) / w.sum()) * w
    return sfft.irfft(L, n, axis=1)[:, :n // 2 + 1] * (n / w.sum())

def show(img, path, title='', extent=None, vmax=None, lines=None, figsize=(14, 6), cmap='RdBu_r'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize); vmax = vmax or np.percentile(np.abs(img), 99.5)
    ax.imshow(img.T, origin='lower', aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax, extent=extent, interpolation='nearest')
    for l in (lines or []): ax.plot(l[0], l[1], l[2] if len(l) > 2 else 'k-', lw=0.6)
    ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)
