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
SOFT_DENSE = 8.0                 # the crusher with plain sound mixed in counts if the values lie this many times denser around the grid values than between them
TWO_PI = 2 * np.pi

def grid(v):
    """The grid of values of one channel: dict(step, offset, share, levels) or None. offset is the grid value nearest
    zero (between -step / 2 and step / 2)."""
    live = v[v != 0.0] if (v != 0.0).any() else v; n = len(live)
    if n < 1000: return None
    lo, hi = float(live.min()), float(live.max())
    if not hi - lo > 1e-9: return None
    A = max(abs(lo), abs(hi)) * (1 + 1e-9); h = 2 * A / HB; H = np.bincount(np.minimum(((live + A) / h).astype(np.int64), HB - 1), minlength=HB).astype(float)
    nf = 4 * HB; F = np.abs(np.fft.rfft(H, nf)) / n                    # F[j]: how well exp(2 pi i u v) lines up over the samples, u = j / (nf h)
    du = 1.0 / (nf * h); j0 = max(int(np.ceil(0.9 / (hi - lo) / du)), 2); j1 = min(int(2 ** (BITS_MAX - 1) / du) + 1, len(F) - 2)       # steps from the whole range down to BITS_MAX
    if j1 - j0 < 8: return None
    # The small samples line up for every u up to one over their size, grid or no grid, and in most sounds they are
    # the many: F is high there anyway. A grid shows as a narrow line on top of that (as wide as one over the range of
    # the values): z = how much of what does not line up anyway lines up at the line, against the lowest F close by
    # on either side.
    W = max(int(np.ceil(1.0 / (hi - lo) / du)), 2); g, sp = 2 * W, 6 * W; Fp = np.concatenate([np.full(sp, F[0]), F, np.full(sp, F[-1])])
    from numpy.lib.stride_tricks import sliding_window_view as swv
    mn = swv(Fp, sp - g + 1).min(1)                                      # mn[i]: lowest of Fp[i .. i + sp - g]
    left = mn[:len(F)]; right = mn[sp + g:sp + g + len(F)]               # lowest of F[j - sp .. j - g] and of F[j + g .. j + sp]
    base = np.maximum(left, right); z = (F - base) / np.maximum(1.0 - base, 1e-9); seg = z[j0:j1 + 1]; top = float(seg.max())
    if top < 0.5: return None
    pk = np.where((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] >= 0.8 * top))[0] + 1
    if not len(pk): return None
    j = int(pk[0]) + j0; j = j - 1 + int(np.argmax(F[j - 1:j + 2])); y0, y1, y2 = F[j - 1], F[j], F[j + 1]; den = y0 - 2 * y1 + y2; step = 1.0 / ((j + (0.5 * (y0 - y2) / den if den < 0 else 0.0)) * du)
    off = float(np.angle(np.exp(1j * TWO_PI * live / step).mean()) / TWO_PI * step)
    for win in (0.25, 0.25, 0.12, 0.06, 0.03, 0.03):                     # the straight line through the values: value = offset + step * (whole number); the window narrows, so that values which only lie around the grid (a crusher that mixes plain sound in) do not pull it
        k = np.round((live - off) / step); r = (live - off) / step - k; on = np.abs(r) < win
        if on.sum() < 100: break
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

def source(n, P, c):
    """For every sample the hold boundary at or before it."""
    if P <= 1.0: return np.arange(n)
    b = np.unique(np.clip(boundaries(n, P, c), 0, n - 1)); src = np.zeros(n, np.int64); src[b] = b; return np.maximum.accumulate(src)

# ---- the crusher that mixes plain sound back in, with soft steps:
#   v = m hold(x) + (1 - m) x,    wet = (1 - m) v + m Q(v)
# Q: the staircase, a whole number of steps on the plateaus and a sine blend between them over the share aa of a step.
# Because of the plain share (1 - m) nothing is lost: the wet sound gives v back (the map is one to one), and v gives x.
def qsoft(u, aa):
    k = np.round(u); fr = u - k
    if aa <= 1e-9: return k
    a1 = (1.0 - aa) / 2.0; sf = np.abs(fr) > a1; out = k.copy(); out[sf] += np.sign(fr[sf]) * 0.5 * (np.sin(np.pi * (np.abs(fr[sf]) - a1) / aa - np.pi / 2) + 1.0); return out
def g0inv(rho, m, aa):
    """fr with (1 - m) fr + m S(fr) = rho (rho between -0.5 and 0.5; S: the blend, 0 on the plateau)."""
    a = np.minimum(np.abs(rho), 0.5); a1 = (1.0 - aa) / 2.0; fr = np.minimum(a / (1.0 - m), a1); bl = a > (1.0 - m) * a1
    if bl.any():
        if aa <= 1e-9: fr[bl] = 0.5
        else:
            lo = np.full(int(bl.sum()), a1); hi = np.full(int(bl.sum()), 0.5); t = a[bl]
            for _ in range(44):
                mid = 0.5 * (lo + hi); up = (1.0 - m) * mid + m * 0.5 * (np.sin(np.pi * (mid - a1) / aa - np.pi / 2) + 1.0) < t; lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
            fr[bl] = 0.5 * (lo + hi)
    return np.sign(rho) * fr
def soft_v(y, step, off, m, aa):
    w = (y - off) / step; K = np.round(w); return off + step * (K + g0inv(w - K, m, aa))
def soft_forward(x, f):
    """x through the crusher f (kind 'soft')."""
    m, aa = f['mix'], f['soft']; v = m * held(x, f['P'], f['c']) + (1.0 - m) * x; o = np.empty_like(x)
    for ch in range(x.shape[1]):
        st = f['step'][min(ch, len(f['step']) - 1)]; of = f['offset'][min(ch, len(f['offset']) - 1)]; o[:, ch] = (1.0 - m) * v[:, ch] + m * (of + st * qsoft((v[:, ch] - of) / st, aa))
    return o
def soft_undo(y, f):
    """The sound before the crusher f (kind 'soft')."""
    m, aa = f['mix'], f['soft']; n = len(y); v = np.empty_like(y)
    for ch in range(y.shape[1]):
        st = f['step'][min(ch, len(f['step']) - 1)]; of = f['offset'][min(ch, len(f['offset']) - 1)]; v[:, ch] = soft_v(y[:, ch], st, of, m, aa)
    if f['P'] <= 1.0: return v
    src = source(n, f['P'], f['c']); x = (v - m * v[src]) / (1.0 - m); isb = src == np.arange(n); x[isb] = v[isb]; return x
def rhythm(y):
    """A hold under plain sound mixed in: the changes are large where a held value is taken and small between. (k, start,
    share): the largest change of every k samples lies at the same place in this share of them."""
    e = np.concatenate([[0.0], (np.diff(y, axis=0) ** 2).sum(1)]); n = len(e); best = (1, 0, 0.0)
    for k in range(2, KMAX + 1):
        r = e[:(n // k) * k].reshape(-1, k); mx = r.max(1); r = r[mx > 1e-18]
        if len(r) < 50: break
        c = np.bincount(np.argmax(r, 1), minlength=k); ph = int(np.argmax(c)); sh = float(c[ph] / len(r))
        if sh > best[2] + (0.02 if k > best[0] else 0.0) or (sh >= best[2] - 0.02 and best[0] > 1 and k % best[0] == 0 and sh >= 0.85): best = (k, ph, sh)
    return best
def rough(x):
    d = x[2:] - 2 * x[1:-1] + x[:-2]; return float((d ** 2).sum())
NBF = 4096                       # bins of a step when the values are folded onto one
def folded(v, st):
    r = v / st; r = r - np.floor(r); return np.bincount(np.minimum((r * NBF).astype(np.int64), NBF - 1), minlength=NBF).astype(float)
def ends(h):
    """The steepest way up and the steepest way down of a folded distribution (smoothed over 3 bins, read around the
    circle): (bin where it goes up, bin where it comes down, how steep both are together)."""
    g = np.convolve(np.concatenate([h[-3:], h, h[:3]]), np.ones(3) / 3.0, 'same')[3:-3]; d = np.roll(g, -2) - np.roll(g, 2)      # d[j]: g[j + 2] - g[j - 2]
    up = int(np.argmax(d)); dn = int(np.argmin(d)); return up, dn, float(d[up] - d[dn])
def sharpen(live, step):
    """The step made exact for values that lie around the grid and not on it. Inside each step the values lean towards
    zero (there are more small samples than large ones), which makes every mean too small; but the stretch around a grid
    value that the crusher fills has sharp ends, at the same place for every grid value. So: the step at which the
    values, folded onto one step, show the sharpest ends."""
    K = np.abs(np.round(live / step))
    for kmax, span, num in ((4, 0.02, 161), (32, 5e-4, 81), (1 << 30, 5e-5, 81), (1 << 30, 2e-6, 41)):
        while kmax < (1 << 30) and int(((K <= kmax) & (K >= 1)).sum()) < 5000 and kmax < K.max(): kmax *= 2          # enough values to see the ends
        v = live[(K <= kmax) & (K >= 1)]
        if len(v) < 500: continue
        cand = step * (1.0 + np.linspace(-span, span, num)); sc = [ends(folded(v, c))[2] for c in cand]; i = int(np.argmax(sc))
        if sc[i] > 1.1 * ends(folded(v, step))[2] or span < 1e-3: step = float(cand[i])                               # (the wide look only moves the step if the ends get clearly sharper)
    return step
def edge(d, hw0):
    """Where the dense part ends, from the distances d (one side, sorted upwards) of the values from the grid value:
    inside the values lie evenly, outside they thin out as 1 / sqrt(1 + (d - hw) / lam) (the blend begins with the
    plateau's own slope). (hw, lam) of the best fit (likelihood) between 0.6 hw0 and 1.6 hw0."""
    lo, hi = 0.6 * hw0, 1.6 * hw0; d = d[(d >= lo) & (d <= hi)]; n = len(d)
    if n < 100: return hw0, 0.0
    best = (-9e99, hw0, 1e-6); span = 0.2 * hw0; ctr = hw0
    for rnd in range(4):
        for hw in ctr + np.linspace(-span, span, 41):
            if not lo < hw < hi: continue
            out = d[np.searchsorted(d, hw):] - hw
            for lam in hw0 * np.logspace(-6, -0.5, 23):
                ll = -0.5 * float(np.log1p(out / lam).sum()) - n * np.log((hw - lo) + 2.0 * lam * (np.sqrt(1.0 + (hi - hw) / lam) - 1.0))
                if ll > best[0]: best = (ll, float(hw), float(lam))
        ctr = best[1]; span *= 0.1
    return best[1], best[2]
def box(live, step):
    """The stretch around each grid value that the values fill: (offset of the grid, half width in steps, share of the
    values inside, how many times denser inside than outside, lam: how fast they thin out beyond the ends), from the
    values of all grid values but the one at zero (silence sits there)."""
    r = live / step; K = np.round(r); v = live[K != 0]
    if len(v) < 500: return None
    h = folded(v, step); up, dn, _ = ends(h); c = ((up + ((dn - up) % NBF) / 2.0) / NBF) % 1.0; hw = ((dn - up) % NBF) / 2.0 / NBF
    if not 2.0 / NBF < hw < 0.25: return None
    lam = 0.0
    for _ in range(2):
        d = ((v / step - c + 0.5) % 1.0) - 0.5; right, l1 = edge(np.sort(d[d >= 0]), hw); left, l2 = edge(np.sort(-d[d < 0]), hw); c = (c + 0.5 * (right - left)) % 1.0; hw = 0.5 * (right + left); lam = 0.5 * (l1 + l2)
    d = np.abs(((v / step - c + 0.5) % 1.0) - 0.5); inside = float((d <= hw).mean()); off = (c + 0.5) % 1.0 - 0.5; dens = inside / max(2 * hw, 1e-9) / max((1.0 - inside) / max(1.0 - 2 * hw, 1e-9), 1e-9)
    return float(off * step), float(hw), inside, float(dens), float(lam)
def soft(y, G):
    """The crusher that mixes plain sound in, read off the sound: dict(mix, soft, step, offset, P, c, ..) or None.
    G: the grids of the channels (grid())."""
    if any(g is None for g in G): return None
    n = len(y); st = []; of = []; hws = []; ins = []; dens = []; lams = []
    for ch in range(y.shape[1]):
        v = y[:, ch]; live = v[v != 0.0]
        if len(live) < 2000: return None
        s1 = sharpen(live, G[ch]['step']); bx = box(live, s1)
        if bx is None: return None
        st.append(s1); of.append(bx[0]); hws.append(bx[1]); ins.append(bx[2]); dens.append(bx[3]); lams.append(bx[4])
    hw = float(np.mean(hws)); inside = float(np.mean(ins)); lam = float(np.mean(lams))
    aa_mass = float(np.clip(1.0 - inside, 0.0, 0.8)); m = float(np.clip(1.0 - 2.0 * hw / max(1.0 - aa_mass, 0.2), 0.3, 0.9995)); aa_lam = aa_mass       # half width (1 - m) (1 - aa) / 2; share inside 1 - aa where the values lie evenly over a step
    for _ in range(6): t = np.pi * np.sqrt(lam * m) / (2.0 * hw); aa_lam = float(t / (1.0 + t)); m = float(np.clip(1.0 - 2.0 * hw / max(1.0 - aa_lam, 0.2), 0.3, 0.9995))                      # lam = aa^2 (1 - m)^2 / (m pi^2): the blend's width from how fast the values thin out beyond the ends
    span = [np.percentile(np.round(y[:, ch] / st[ch]), [1, 99]) for ch in range(y.shape[1])]; fine = min(float(b - a) for a, b in span) >= 40.0
    aa = aa_mass if fine else aa_lam; m = float(np.clip(1.0 - 2.0 * hw / max(1.0 - aa, 0.2), 0.3, 0.9995))            # the share where the values spread over many steps, the thinning where over few
    k, ph, sh = rhythm(y); P, c = (float(k), float(ph)) if sh >= 0.85 else (1.0, 0.0)
    if P > 1.0:
        # With a hold the blend can be made exact: the half width ties mix and blend together, and of the pairs it
        # allows the right one leaves no trace of the hold in the undone sound: its second difference is as large
        # across the hold's boundaries as between them (any other pair leaves teeth, and the boundaries stand out).
        yy = y if n <= 3 * SR else np.vstack([y[i:i + SR // 2] for i in (np.linspace(0, n - SR // 2, 6).astype(int) // int(P)) * int(P)]); src = source(len(yy), P, c); isb = src == np.arange(len(yy)); tb = isb[1:-1] | isb[2:]
        def teeth(a):
            o = soft_undo(yy, dict(mix=1.0 - 2.0 * hw / (1.0 - a), soft=a, step=st, offset=of, P=P, c=c)); e = ((o[2:] - 2 * o[1:-1] + o[:-2]) ** 2).sum(1); return float(e[tb].mean() / max(e[~tb].mean(), 1e-300))
        base = teeth(aa); best = (base, aa); span = 0.06
        for rnd in range(3):
            for a in best[1] + np.linspace(-span, span, 13):
                if 0.0 <= a <= 0.8 and 1.0 - 2.0 * hw / (1.0 - a) >= 0.3:
                    t = teeth(float(a))
                    if t < best[0]: best = (t, float(a))
            span /= 6.0
        if best[0] < 0.98 * base: aa = best[1]; m = float(np.clip(1.0 - 2.0 * hw / (1.0 - aa), 0.3, 0.9995))
    return dict(kind='soft', mix=m, soft=aa, step=st, offset=of, P=P, c=c, rhythm=sh, half=hw, inside=inside, aa_mass=aa_mass, aa_lam=aa_lam, lam=lam, dense=float(np.min(dens)), bits=float(np.mean([np.log2(2.0 / v) for v in st])))

def find(y):
    """dict(found, kind ('bare': the values lie on the grid; 'soft': plain sound mixed in), step and offset per
    channel, bits, P, c, ..)."""
    out = dict(found=False, n=len(y))
    if len(y) < SR // 4: return out
    G = [grid(y[:, ch]) for ch in range(y.shape[1])]; H = hold(y); out.update(grids=G, hold=H)
    ok = [g is not None and g['share'] >= SHARE and g['bits'] <= BITS_MAX for g in G]; hasHold = H['P'] >= 2.0 and H['share'] >= 0.99 and H['changes'] >= 50
    if not (all(ok) or hasHold):
        f = soft(y, G); out['soft_try'] = f
        if f is not None and f['dense'] >= SOFT_DENSE and f['inside'] >= 0.3 and f['half'] <= 0.2 and f['bits'] <= BITS_MAX: out.update(f); out['found'] = True
        return out
    out.update(found=True, kind='bare', step=[g['step'] if o else 0.0 for g, o in zip(G, ok)], offset=[g['offset'] if o else 0.0 for g, o in zip(G, ok)], bits=float(np.mean([g['bits'] for g, o in zip(G, ok) if o])) if any(ok) else 24.0, P=H['P'] if hasHold else 1.0, c=H['c'] if hasHold else 0.0)
    return out
def onto(z, f, s=1.0):
    """The second sound through the same crusher: the same grid of values and the same hold. s: Amount (1 = as found)."""
    if not f.get('found') or s < 1e-12: return z.copy()
    if f.get('kind') == 'soft': q = soft_forward(z, f); return q if abs(s - 1.0) < 1e-12 else z + s * (q - z)
    q = z.copy()
    for ch in range(z.shape[1]):
        st = f['step'][min(ch, len(f['step']) - 1)]; of = f['offset'][min(ch, len(f['offset']) - 1)]
        if st > 0: q[:, ch] = of + st * np.round((z[:, ch] - of) / st)
    q = held(q, f['P'], f['c'])
    return q if abs(s - 1.0) < 1e-12 else z + s * (q - z)

def undo(y, f):
    """The sound before the crusher, where it can be had: exactly for the kind 'soft' (None otherwise)."""
    return soft_undo(y, f) if f.get('found') and f.get('kind') == 'soft' else None
