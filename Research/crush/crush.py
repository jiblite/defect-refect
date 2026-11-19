# The crush engine (BITCRUSH; reference for Source/Core/Crush.h): a sound whose samples were rounded to a grid of
# values (fewer bits) and held for several samples each (a lower sample rate).
#   wet[t] = Q(dry[b(t)]),  Q(v) = offset + step * round((v - offset) / step),  b(t) = the last hold boundary at or before t
# Read off blind:
#   1. the grid: a sound on a grid of values has an amplitude distribution made of spikes a step apart, and the
#      Fourier transform of that distribution (how well exp(2 pi i u v) lines up over all samples) has a line at
#      u = 1 / step. The step and the offset are then made exact by a straight-line fit through the values themselves;
#   2. the hold: the places where the sound changes at all. With a hold of k samples they all lie k apart or a multiple
#      of it; with a hold that is not a whole number (a sample rate that does not divide the file's) they lie on the
#      pattern floor/ceil of a straight line, whose slope and start are fitted.
# Put onto a second sound: the same grid of values (in absolute terms, as the same crusher would) and the same hold
# pattern.
import numpy as np
SR = 48000
BITS_MAX = 13.0                  # a grid finer than this many bits (against full scale, 2 / step) is an ordinary file, not a crush
ON = 0.02                        # a sample is on the grid if it is within this share of a step of a grid value
SHARE = 0.995                    # .. and this share of the live samples has to be
HB = 1 << 18                     # bins of the amplitude histogram
KMAX = 256                       # the longest hold looked for (samples)
TWO_PI = 2 * np.pi

def grid(v):
    """The grid of values of one channel: dict(step, offset, share, levels) or None. offset is the grid value nearest
    zero (between -step / 2 and step / 2)."""
    live = v[v != 0.0] if (v != 0.0).any() else v; n = len(live)
    if n < 1000: return None
    lo, hi = float(live.min()), float(live.max())
    if not hi - lo > 1e-9: return None
    A = max(abs(lo), abs(hi)) * (1 + 1e-9); h = 2 * A / HB; H = np.bincount(np.minimum(((live + A) / h).astype(np.int64), HB - 1), minlength=HB).astype(float)
    nf = 4 * HB; F = np.abs(np.fft.rfft(H, nf)) / n                    # F[j]: how well exp(2 pi i u v) lines up, u = j / (nf h)
    du = 1.0 / (nf * h); j0 = max(int(np.ceil(0.9 / (hi - lo) / du)), 2); j1 = min(int(2 ** (BITS_MAX - 1) / du) + 1, len(F) - 2)       # steps from the whole range down to BITS_MAX
    if j1 - j0 < 8: return None
    seg = F[j0:j1 + 1]; top = float(seg.max())
    if top < 0.5: return None
    pk = np.where((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] >= 0.9 * top))[0] + 1
    if not len(pk): return None
    j = int(pk[0]) + j0; y0, y1, y2 = F[j - 1], F[j], F[j + 1]; den = y0 - 2 * y1 + y2; step = 1.0 / ((j + (0.5 * (y0 - y2) / den if den < 0 else 0.0)) * du)
    off = 0.0
    for _ in range(4):                                                   # the straight line through the values: value = offset + step * (whole number)
        ph = np.angle(np.exp(1j * TWO_PI * live / step).mean()) / TWO_PI; off = ph * step; k = np.round((live - off) / step); r = (live - off) / step - k; on = np.abs(r) < 0.25
        if on.sum() < 100: return None
        kk = k[on]; vv = live[on]; km = kk.mean(); den = float(((kk - km) ** 2).sum())
        if not den > 0: return None
        step = float(((kk - km) * (vv - vv.mean())).sum() / den); off = float(vv.mean() - step * km)
    off = off - step * np.round(off / step); k = np.round((live - off) / step); r = (live - off) / step - k
    return dict(step=step, offset=off, share=float((np.abs(r) < ON).mean()), levels=int(k.max() - k.min()) + 1, bits=float(np.log2(2.0 / step)))

def hold(y):
    """The hold pattern from the places where the sound changes: dict(P: samples a held value lasts (1: no hold), c: the
    pattern's start, share: how many of the changes lie on it). The boundaries are ceil(j P + c), j = 0, 1, .."""
    n = len(y); ch = np.where((np.diff(y, axis=0) != 0).any(1))[0] + 1
    none = dict(P=1.0, c=0.0, share=1.0, changes=int(len(ch)))
    if len(ch) < 50: return none
    best = None
    for k in range(KMAX, 1, -1):                                         # a whole number of samples: the longest that fits
        cnt = np.bincount(ch % k, minlength=k); ph = int(np.argmax(cnt))
        if cnt[ph] >= 0.99 * len(ch): best = (float(k), float(ph), float(cnt[ph] / len(ch))); break
    if best is not None: return dict(P=best[0], c=best[1], share=best[2], changes=int(len(ch)))
    gaps = np.diff(ch); g = int(np.percentile(gaps, 2))
    if g < 2: return none                                                # changes one sample apart: no hold
    # not a whole number: the line in the spectrum of the changes, then a straight-line fit of position against count
    nf = 1 << int(np.ceil(np.log2(n))); c = np.zeros(nf); c[ch] = 1.0; S = np.abs(np.fft.rfft(c - c[:n].mean() * (np.arange(nf) < n))); lo = int(nf / (g + 1.5)); hi = int(nf / max(g - 0.5, 1.5)) + 1
    j = lo + int(np.argmax(S[lo:hi + 1])); y0, y1, y2 = S[j - 1], S[j], S[j + 1]; den = y0 - 2 * y1 + y2; P = nf / (j + (0.5 * (y0 - y2) / den if den < 0 else 0.0))
    for _ in range(3):
        idx = np.concatenate([[0], np.cumsum(np.round(gaps / P))]); im = idx.mean(); den = float(((idx - im) ** 2).sum())
        if not den > 0: return none
        P = float(((idx - im) * (ch - ch.mean())).sum() / den)
    # the start: ceil(j P + c) = position  <=>  position - 1 - j P < c <= position - j P, for every change; P moved a hair so that the room for c is widest
    def room(P): r = ch - idx * P; return float(r.min()), float(r.max() - 1.0)
    a, b = P * (1 - 2e-5), P * (1 + 2e-5)
    for _ in range(60):
        p1 = a + (b - a) / 3; p2 = b - (b - a) / 3; h1, l1 = room(p1); h2, l2 = room(p2)
        if h1 - l1 < h2 - l2: a = p1
        else: b = p2
    P = 0.5 * (a + b); hi_, lo_ = room(P); c0 = 0.5 * (hi_ + lo_); on = np.ceil(idx * P + c0) == ch
    if not P >= 2.0: return none
    return dict(P=P, c=c0, share=float(on.mean()), changes=int(len(ch)), room=hi_ - lo_)

def boundaries(n, P, c):
    """The hold boundaries in 0 .. n - 1 (always with one at or before sample 0)."""
    j0 = int(np.floor(-c / P)) - 1; j1 = int(np.ceil((n - c) / P)) + 1; b = np.ceil(np.arange(j0, j1 + 1) * P + c - 1e-9).astype(np.int64)
    return b
def held(q, P, c):
    """q with every sample replaced by the one at the last hold boundary at or before it."""
    n = len(q)
    if P <= 1.0: return q.copy()
    b = boundaries(n, P, c); b = np.unique(np.clip(b, 0, n - 1)); src = np.zeros(n, np.int64); src[b] = b; src = np.maximum.accumulate(src); return q[src]

def find(y):
    """dict(found, step and offset per channel, bits, P, c, ..)."""
    out = dict(found=False, n=len(y))
    if len(y) < SR // 4: return out
    G = [grid(y[:, ch]) for ch in range(y.shape[1])]; H = hold(y); out.update(grids=G, hold=H)
    ok = [g is not None and g['share'] >= SHARE and g['bits'] <= BITS_MAX for g in G]; hasHold = H['P'] >= 2.0 and H['share'] >= 0.99 and H['changes'] >= 50
    if not (all(ok) or hasHold): return out
    out.update(found=True, step=[g['step'] if o else 0.0 for g, o in zip(G, ok)], offset=[g['offset'] if o else 0.0 for g, o in zip(G, ok)], bits=float(np.mean([g['bits'] for g, o in zip(G, ok) if o])) if any(ok) else 24.0, P=H['P'] if hasHold else 1.0, c=H['c'] if hasHold else 0.0)
    return out
def onto(z, f, s=1.0):
    """The second sound through the same crusher: the same grid of values and the same hold. s: Amount (1 = as found)."""
    if not f.get('found') or s < 1e-12: return z.copy()
    q = z.copy()
    for ch in range(z.shape[1]):
        st = f['step'][min(ch, len(f['step']) - 1)]; of = f['offset'][min(ch, len(f['offset']) - 1)]
        if st > 0: q[:, ch] = of + st * np.round((z[:, ch] - of) / st)
    q = held(q, f['P'], f['c'])
    return q if abs(s - 1.0) < 1e-12 else z + s * (q - z)
