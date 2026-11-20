# Defect Refect, the declipper (prototype of Source/Core/Declip.h): the way back for sounds whose tops were cut flat.
# Each half-wave of the bent sound (its values above zero, or those below) is looked at by itself:
#   hard   many values sit exactly at its top and the values just under them are no denser than further down: a plain
#          cut. Nothing under the cut is touched; the values at the cut are unknown beyond it and are rebuilt.
#   soft   a pile of values under the top with a knee below it, and the meter has found a bend: the meter's reading is
#          walked back up to a first height, the rest is rebuilt; the rebuilt values then tell how the curve runs above
#          the first height, the lower part of that zone is walked back along this curve and only the very top is rebuilt
#          again.
#   none   no pile: the bend engine's careful way back (bengine.py), nothing rebuilt.
# Rebuilding is social sparsity with a persistent empirical Wiener shrinkage (Siedenburg, Kowalski, Doerfler 2014): the
# answer is looked for among sounds that agree with the known values, stay beyond the bounds where the values are not
# known, and need few cells of a time-frequency picture, where a cell counts with its neighbours in time.
# Two tests the engine runs on the sound itself keep it from doing harm: the rebuilding is tried one step down (the trusted
# values cut again 3 dB lower and rebuilt: a dense, noise-like sound fails), and a second pass that fills the picture more
# densely than the first is thrown away (the flat tops were not a curve's work).
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bend'))
import numpy as np, bfeat, bengine
# ---------------------------------------------------------------- the numbers (the same in Source/Core/Declip.h)
N0 = 2048                                                    # the picture's window at 44.1 and 48 kHz; the hop is half of it
SPAN = 3; TOL = 1e-4; LAM_HI = -0.5; LAM_LO = -3.5           # neighbours in time; when a step of the schedule has settled; the schedule (powers of ten of the picture's top)
MARGIN = 3; CORE = 192                                       # hops of known sound kept around what is rebuilt; hops rebuilt in one go at most
TIE_DB = 0.01; LIVE_DB = -60.0; TIE_SHARE = 0.002; TIE_N = 16; HARD_RATIO = 1.3; PILE_SHARE = 0.01; PILE_RATIO = 1.0
SLOPE = 0.5; KEEP = 0.8; BINS = 300; LEAST = 64; SMOOTH = 2; ZONE_MIN = 256; LOW_MIN = 128
TEST_DB = 3.0; TEST_MIN = 20.0; TEST_HOPS = 144; DENSE_MAX = 1.2
FIRST = 300                                                  # hops per channel that a soft top's first pass works on (a longer sound: its stretches with the most unknown values)
SURE_HARD = 0.9; TOP_HARD = 0.5                              # a plain cut shows by itself: how sure, and from how many dB of rebuilt peak
STEP = 1.0 / 16; L0 = -96.0; L1 = 12.0                       # the curves' tables, as bengine.Curve's
OPT = dict(bud_hard=(8, 12), bud_first=(16, 25), bud_second=(8, 12), core=CORE, test=True, dense=True, first=FIRST)       # rounds: (steps of the schedule, rounds per step) for a plain cut and for a soft top's two passes
def window(sr):
    """The picture's window for a sound at sr."""
    return int(min(max(N0 * 2 ** int(np.floor(np.log2(sr / 48000.0) + 0.5)), 512), 8192))
# ---------------------------------------------------------------- the picture
class Gabor:
    """A tight frame: windows of N samples every N / 2, a sine window scaled so that going to the picture and back gives
    the sound again and the picture's energy is the sound's. N samples of nothing are added before and after the sound."""
    def __init__(s, n, N):
        s.n = n; s.N = N; s.H = N // 2; s.w = np.sin(np.pi * (np.arange(N) + 0.5) / N); s.T = (n + N) // s.H + 2; s.len = (s.T - 1) * s.H + N
    def fwd(s, x):
        buf = np.zeros(s.len); buf[s.N:s.N + s.n] = x; fr = np.lib.stride_tricks.sliding_window_view(buf, s.N)[::s.H][:s.T]
        return np.fft.rfft(fr * s.w, axis=1) / np.sqrt(s.N)
    def back(s, A):
        fr = np.fft.irfft(A, s.N, axis=1) * np.sqrt(s.N) * s.w; buf = np.zeros(s.len)
        for k in (0, 1): part = fr[k::2]; buf[k * s.H:k * s.H + part.size] += part.ravel()
        return buf[s.N:s.N + s.n]
def energy(A):
    """A picture's energy as the sound's: the inner bins count twice (the half of the spectrum that is not kept)."""
    P = np.abs(A) ** 2; return P[:, 0].sum() + 2 * P[:, 1:-1].sum() + P[:, -1].sum()
def shrink(A, lam, span=SPAN):
    """Persistent empirical Wiener: a cell is kept by 1 - lam^2 / (the mean energy of its neighbours in time), never under 0."""
    P = np.abs(A) ** 2; Pp = np.pad(P, ((span // 2, span // 2), (0, 0)), mode='edge'); c = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), Pp]), 0); P = (c[span:] - c[:-span]) / span
    return A * np.maximum(1.0 - lam * lam / np.maximum(P, 1e-30), 0.0)
def top_of(A):
    """The level the schedule hangs on: 999 of 1000 of the picture's cells are weaker (read off a table in steps of 0.05 dB)."""
    P = np.abs(A).ravel() ** 2; i = np.clip(np.floor((10 * np.log10(np.maximum(P, 1e-30)) + 240.0) / 0.05).astype(int), 0, 5999); c = np.cumsum(np.bincount(i, minlength=6000))
    k = int(np.searchsorted(c, 0.999 * len(P))); return float(np.sqrt(10 ** ((-240.0 + 0.05 * (k + 1)) / 10)))
def dense(u, N=N0, off=None):
    """How densely a sound fills its picture: the sum of the cells' sizes over the root of their energy (both channels).
    off: per channel stretches (a, b) of the sound; the frames that touch one of them do not count."""
    a = 0.0; b = 0.0; H = N // 2
    for c in range(u.shape[1]):
        A = np.abs(Gabor(len(u), N).fwd(u[:, c]))
        if off is not None and len(off[c]):
            t = np.arange(len(A)); keep = np.ones(len(A), bool)
            for ca, cb in off[c]: keep &= ~(((t - 2) * H < cb) & (t * H > ca))
            A = A[keep]
        a += A.sum(); b += (A ** 2).sum()
    return float(a / max(np.sqrt(b), 1e-30))
# ---------------------------------------------------------------- rebuilding
def solve(y, rel, lo, hi, lams, inner, N=N0):
    """One stretch of one channel. y: the values where rel says they are known. Elsewhere the value is only known to lie at
    or above lo (where lo is finite) or at or below hi (where hi is finite). -> the stretch rebuilt (known values as they are)."""
    G = Gabor(len(y), N); up = ~rel & np.isfinite(lo); dn = ~rel & np.isfinite(hi); lo = np.where(up, lo, 0.0); hi = np.where(dn, hi, 0.0)
    A = G.fwd(np.where(rel, y, np.where(up, lo, hi))); Z = A.copy(); t = 1.0
    for lam in lams:
        for it in range(inner):
            x = G.back(Z); g = np.where(rel, y - x, 0.0) + np.where(up, np.maximum(lo - x, 0.0), 0.0) + np.where(dn, np.minimum(hi - x, 0.0), 0.0)      # what is missing towards the known values and the bounds
            An = shrink(Z + G.fwd(g), lam); tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)); Z = An + ((t - 1.0) / tn) * (An - A)
            d = np.sqrt(energy(An - A) / max(energy(An), 1e-30)); A = An; t = tn
            if d < TOL: break
    x = G.back(A); out = np.where(rel, y, x); out = np.where(up, np.maximum(out, lo), out); return np.where(dn, np.minimum(out, hi), out)
def pieces(unknown, H, core=CORE):
    """What is rebuilt in one go: (a, b, ca, cb), the stretch [a, b) and the part [ca, cb) of it whose unknown values are taken.
    Unknown values less than 2 * MARGIN hops apart belong together; a long run is cut into equal parts of at most `core` hops,
    each solved with MARGIN hops of its surroundings."""
    idx = np.flatnonzero(unknown); n = len(unknown); M = MARGIN * H; out = []
    if len(idx) == 0: return out
    cut = np.flatnonzero(np.diff(idx) > 2 * M); s = np.concatenate([[idx[0]], idx[cut + 1]]); e = np.concatenate([idx[cut], [idx[-1]]]) + 1
    for a, b in zip(s, e):
        a = int(a); b = int(b); k = -(-(b - a) // (core * H)); L = -(-(b - a) // k)
        for j in range(k): ca = a + j * L; cb = min(b, ca + L); out.append((max(0, ca - M), min(n, cb + M), ca, cb))
    return out
def rebuild(x, rel, lo, hi, bud, N=N0, core=CORE, only=None, into=None):
    """x (n, ch) with its unknown values (~rel) rebuilt beyond their bounds. bud: (steps of the schedule, rounds per step).
    only: per channel the numbers of the pieces that are worked on (None: all); into: the sound they are put into (None: x)."""
    out = x.copy() if into is None else into.copy()
    for c in range(x.shape[1]):
        r = rel[:, c]
        if r.all(): continue
        y = np.where(r, x[:, c], np.where(np.isfinite(lo[:, c]), lo[:, c], hi[:, c])); lams = top_of(Gabor(len(y), N).fwd(y)) * 10 ** np.linspace(LAM_HI, LAM_LO, bud[0])
        for i, (a, b, ca, cb) in enumerate(pieces(~r, N // 2, core)):
            if only is not None and i not in only[c]: continue
            o = solve(y[a:b], r[a:b], lo[a:b, c], hi[a:b, c], lams, bud[1], N); m = ~r[ca:cb]; out[ca:cb, c][m] = o[ca - a:cb - a][m]
    return out
def choose(rel, H, core=CORE, first=FIRST):
    """A long sound's first pass: per channel the pieces with the most unknown values (among equals the earlier one), until
    `first` hops are covered. -> only (per channel the numbers of the chosen pieces), seen (where a chosen piece's own part
    lies), off (per channel the other pieces' own parts), whole (all pieces are chosen: any sound of up to `first` hops)."""
    only = []; off = []; seen = np.zeros(rel.shape, bool); whole = True
    for c in range(rel.shape[1]):
        P = pieces(~rel[:, c], H, core); cnt = [int((~rel[ca:cb, c]).sum()) for a, b, ca, cb in P]; got = 0; s = set()
        for i in sorted(range(len(P)), key=lambda i: (-cnt[i], i)):
            if got >= first * H: break
            s.add(i); got += P[i][3] - P[i][2]; seen[P[i][2]:P[i][3], c] = True
        only.append(s); off.append([(P[i][2], P[i][3]) for i in range(len(P)) if i not in s]); whole = whole and len(s) == len(P)
    return only, seen, off, whole
# ---------------------------------------------------------------- what kind of top a half-wave has
def stats(y, vp):
    """Per half-wave: c its top value; tn, t the number and the share of its live values within TIE_DB of the top; p25 the
    share within 0.25 dB of its peak vp; R the density of values (per dB) just under the ties against that 1 to 2 dB under
    the top; P the density in the top quarter dB under the peak (ties and all) against that 1 to 2 dB under the peak."""
    out = []
    for h, pol in ((0, 1.0), (1, -1.0)):
        v = pol * y[pol * y > vp[h] * 10 ** (LIVE_DB / 20)]; n = max(len(v), 1); c = float(v.max()) if len(v) else 0.0
        lv = 20 * np.log10(np.maximum(v, 1e-30) / max(c, 1e-30)); lr = 20 * np.log10(np.maximum(v, 1e-30) / max(vp[h], 1e-30)); tn = int(np.sum(lv >= -TIE_DB))
        d_top = np.sum((lv >= -0.25) & (lv < -TIE_DB)) / n / (0.25 - TIE_DB); d_low = np.sum((lv >= -2.0) & (lv < -1.0)) / n; p25 = float(np.sum(lr >= -0.25)) / n; p_low = np.sum((lr >= -2.0) & (lr < -1.0)) / n
        out.append(dict(c=c, n=n, tn=tn, t=tn / n, p25=p25, R=float(d_top / max(d_low, 1e-9)), P=float((p25 / 0.25) / max(p_low, 1e-9))))
    return out
def tied(st): return st['tn'] >= TIE_N and st['t'] >= TIE_SHARE
def regime(st, kept):
    if tied(st) and st['R'] <= HARD_RATIO: return 'hard'
    if kept and st['p25'] >= PILE_SHARE and st['P'] >= PILE_RATIO: return 'soft'
    return 'none'
# ---------------------------------------------------------------- the reading as a curve over the bent sound's levels
def out_side(S):
    """One half-wave's reading over the bent sound's own levels: (o, q), o = level of the bent value in dB under the
    half-wave's peak (rising), q = by how many dB it was squeezed. The reading may not rise faster than the level."""
    GRID = bfeat.GRID; S = np.maximum.accumulate(np.clip(np.asarray(S, float), 0.0, bengine.MAXSQ)); r = S.copy()
    for i in range(1, len(r)): r[i] = min(r[i], r[i - 1] + (GRID[i] - GRID[i - 1]))
    lev = np.concatenate([[-90.0, bfeat.SMALL - 3.0], GRID]); sq = np.concatenate([[0.0, 0.0], r]); o = lev - sq + sq[-1]
    for i in range(1, len(o)): o[i] = max(o[i], o[i - 1] + 1e-6)
    return o, sq
def trust_at(o, q):
    """The height (dB under the peak) up to which the reading is walked back: where the bent value still moves at least SLOPE
    dB per dB of the value before the bend. None: all the way up."""
    lev = o + q - q[-1]; s = np.diff(o) / np.maximum(np.diff(lev), 1e-9); i = len(s)
    while i > 0 and s[i - 1] < SLOPE: i -= 1
    return None if i == len(s) else float(o[i])
def setup(y, S, vp, kinds, st):
    """The first pass's start: x (the values walked back where they are trusted), rel (where that is), lo and hi (the bounds
    elsewhere), and per half-wave what the walk was: None, or b (the bound), yb (the bent value there), and for a soft
    half-wave the reading's curve (o, q) and the first height os in dB."""
    x = np.zeros_like(y); rel = np.ones(y.shape, bool); lo = np.full(y.shape, -np.inf); hi = np.full(y.shape, np.inf); first = [None, None]; kinds = list(kinds)
    for h, pol in ((0, 1.0), (1, -1.0)):
        m = pol * y > 0; v = np.maximum(pol * y, 0.0); f = None
        if kinds[h] == 'soft':
            o, q = out_side(S[h]); os_ = trust_at(o, q)
            if tied(st[h]): os_ = min(20 * np.log10(st[h]['c'] / vp[h]) - TIE_DB, 0.0 if os_ is None else os_)
            if os_ is None: kinds[h] = 'none'                                       # no height from which the reading is flat: nothing to rebuild
            else:
                lv = 20 * np.log10(np.maximum(v, 1e-30) / vp[h]); xb = v * 10 ** (np.interp(np.minimum(lv, os_), o, q) / 20); f = m & (lv >= os_)
                b = vp[h] * 10 ** ((os_ + float(np.interp(os_, o, q))) / 20); first[h] = dict(b=b, yb=vp[h] * 10 ** (os_ / 20), o=o, q=q, os=os_)
        if kinds[h] == 'hard': b = st[h]['c'] * 10 ** (-TIE_DB / 20); f = m & (v >= b); xb = v; first[h] = dict(b=b, yb=b)
        elif kinds[h] == 'none': xb = bengine.Curve(bengine.careful(S[h]), vp[h], bengine.SPREAD).back(v)
        x = np.where(m, pol * xb, x)
        if f is not None and f.any():
            rel &= ~f
            if pol > 0: lo = np.where(f, b, lo)
            else: hi = np.where(f, -b, hi)
    return x, rel, lo, hi, first, kinds
# ---------------------------------------------------------------- the curve the rebuilt values tell
def pava(v, w):
    """The closest rising row of values to v (weights w)."""
    val = []; wt = []; cnt = []
    for a, b in zip(v, w):
        val.append(a); wt.append(b); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            val[-2] = (val[-2] * wt[-2] + val[-1] * wt[-1]) / (wt[-2] + wt[-1]); wt[-2] += wt[-1]; cnt[-2] += cnt[-1]; val.pop(); wt.pop(); cnt.pop()
    return np.repeat(val, cnt)
def curve_from(yy, xx, y0, x0):
    """The curve from (y0, x0) on, read from pairs (bent value, rebuilt value), both positive: nodes (ym, xm), both rising.
    The pairs in order of the bent value, in up to BINS groups of equal size (LEAST pairs at least); per group the middle
    values; the row made rising, smoothed over SMOOTH groups to each side, made rising again."""
    o = np.argsort(yy, kind='stable'); yy = yy[o]; xx = xx[o]; n = len(yy); nb = int(max(1, min(BINS, n // LEAST))); ym = [y0]; xm = [x0]; w = [1e9]
    for k in range(nb):
        a = (k * n) // nb; b = ((k + 1) * n) // nb
        if b > a: ym.append(float(np.median(yy[a:b]))); xm.append(float(np.median(xx[a:b]))); w.append(float(b - a))
    ym = np.array(ym); xm = pava(np.maximum(np.array(xm), x0), np.array(w))
    if len(xm) > 2 * SMOOTH + 1:
        s = np.convolve(np.pad(xm, SMOOTH, mode='edge'), np.ones(2 * SMOOTH + 1) / (2 * SMOOTH + 1), 'valid'); s[0] = x0; xm = np.maximum.accumulate(s)
    keep = np.concatenate([[True], np.diff(ym) > 0]); return ym[keep], xm[keep]
def quantile(v, p):
    s = np.sort(v); t = p * (len(s) - 1); i = int(np.floor(t)); j = min(i + 1, len(s) - 1); return float(s[i] + (t - i) * (s[j] - s[i]))
def second(y, x, rel, lo, hi, out0, kinds, st, seen=None):
    """The second pass's start. For the soft half-waves the curve above the first height is read from the first rebuild out0
    (from the zone's lower KEEP, never from the ties), those values are walked back along it and count as known; the rest of
    the zone stays unknown above the curve's last value. seen: where the first pass has worked (None: everywhere); the
    curve is read there and serves the whole sound. -> x, rel, lo, hi, per half-wave None or (ym, xm, th), did."""
    x1 = x.copy(); rel1 = rel.copy(); lo1 = lo.copy(); hi1 = hi.copy(); cur = [None, None]
    for h, pol in ((0, 1.0), (1, -1.0)):
        zone = ~rel & (pol * y > 0); zs = zone if seen is None else zone & seen
        if kinds[h] != 'soft' or zs.sum() < ZONE_MIN: continue
        yy = pol * y[zs]; xx = pol * out0[zs]; th = quantile(yy, KEEP)
        if tied(st[h]): th = min(th, st[h]['c'] * 10 ** (-TIE_DB / 20))
        low = yy < th
        if low.sum() < LOW_MIN: continue
        ym, xm = curve_from(yy[low], xx[low], float(yy.min()), float(xx.min())); mid = zone & (pol * y < th); top = zone & ~mid; x1 = np.where(mid, pol * np.interp(pol * y, ym, xm), x1); rel1 |= mid
        if pol > 0: lo1 = np.where(top, xm[-1], lo1); lo1 = np.where(mid, -np.inf, lo1)
        else: hi1 = np.where(top, -xm[-1], hi1); hi1 = np.where(mid, np.inf, hi1)
        cur[h] = (ym, xm, th)
    return x1, rel1, lo1, hi1, cur, cur[0] is not None or cur[1] is not None
# ---------------------------------------------------------------- the test on the sound itself
def picture(x, sr=48000, N=N0, bands=48, f_lo=30.0, f_hi=20000.0):
    """A plain picture of a sound: power per frame (a window of N every N / 4) in `bands` bands of equal width on a mel
    scale, both channels after one another. (frames, bands)."""
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N); mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0); inv = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    e = np.clip(np.floor(inv(np.linspace(mel(f_lo), mel(min(f_hi, 0.5 * sr)), bands + 1)) * N / sr + 0.5).astype(int), 1, N // 2); P = []
    for c in range(x.shape[1]):
        fr = np.lib.stride_tricks.sliding_window_view(x[:, c], N)[::N // 4]; S = np.abs(np.fft.rfft(fr * w, axis=1)) ** 2; cs = np.concatenate([np.zeros((len(S), 1)), np.cumsum(S, 1)], 1)
        P.append(np.stack([cs[:, max(b, a + 1)] - cs[:, a] for a, b in zip(e[:-1], e[1:])], 1))
    return np.concatenate(P, 0)
def picdist(PA, PB, floor_db=-60.0, cap_db=30.0):
    """How far picture PB is from PA (the reference): the mean difference in dB after the best level match, cells under
    floor_db of the reference's loudest counting as the floor, no cell for more than cap_db."""
    LA = 10 * np.log10(np.maximum(PA, (PA.max() + 1e-20) * 10 ** (floor_db / 10))); LB = 10 * np.log10(np.maximum(PB, 1e-30)); live = LA > LA.min() + 6
    g = float(np.median((LA - LB)[live])) if live.any() else 0.0; return float(np.minimum(np.abs(LA - np.maximum(LB + g, LA.min())), cap_db).mean())
def selftest(x, rel, lo, hi, bud, sr=48000, N=N0, core=CORE):
    """The rebuilding tried on the sound itself, one step down: the trusted values are cut again TEST_DB under each half-wave's
    bound, rebuilt from below, and held against what was there. 100 = the rebuilt band matches, 0 = no better than the cut,
    below 0 = worse. (Above the real bound the rebuilt values are cut at the bound, as the sound in hand is.) Tried on the
    TEST_HOPS hops of the sound that hold the most values above the test's cut, with the rounds the rebuilding itself has."""
    bp = float(np.min(lo[np.isfinite(lo)])) if np.isfinite(lo).any() else np.inf; bn = float(np.max(hi[np.isfinite(hi)])) if np.isfinite(hi).any() else -np.inf
    held = np.where(rel, x, np.where(np.isfinite(lo), lo, hi)); k = 10 ** (-TEST_DB / 20); cp = bp * k; cn = bn * k; relv = (held < cp) & (held > cn); H = N // 2; nh = len(x) // H
    if nh > TEST_HOPS:
        cs = np.concatenate([[0], np.cumsum((~relv[:nh * H]).reshape(nh, -1).sum(1))]); a = int(np.argmax(cs[TEST_HOPS:] - cs[:-TEST_HOPS])) * H; held = held[a:a + TEST_HOPS * H]; relv = relv[a:a + TEST_HOPS * H]
    cut = np.clip(held, cn, cp); out = np.clip(rebuild(cut, relv, np.where(held >= cp, cp, -np.inf), np.where(held <= cn, cn, np.inf), bud, N, core), bn, bp)
    PA = picture(held, sr, N); d0 = picdist(PA, picture(cut, sr, N)); return 100.0 * (1.0 - picdist(PA, picture(out, sr, N)) / max(d0, 1e-9)), d0
# ---------------------------------------------------------------- a rebuilt half-wave's curve, on a table
class Tab:
    """A half-wave's curve from nodes (value before the bend xn -> bent value yn, both rising): the line y = x under the
    first node, straight pieces between the nodes, flat above the last. On a table as bengine.Curve's, around `ref`."""
    def __init__(s, ref, xn, yn, xp, vp):
        l = np.arange(L0, L1 + 1e-4, STEP); s.inn = ref * 10 ** (l / 20); s.out = np.where(s.inn < xn[0], s.inn, np.interp(s.inn, xn, yn)); s.out = np.maximum.accumulate(s.out)
        s.xp = float(xp); s.vp = float(vp); s.top = float(max(20 * np.log10(max(xp, 1e-30) / max(vp, 1e-30)), 0.0))
    def fwd(s, x): x = np.asarray(x, float); return np.where(x < s.inn[0], x, np.interp(x, s.inn, s.out))
def rising(xn, yn):
    """Nodes with x and y both rising: a node that does not lie further out than the one before it is left out."""
    X = [xn[0]]; Y = [yn[0]]
    for a, b in zip(xn[1:], yn[1:]):
        if a > X[-1] * (1.0 + 1e-9) and b >= Y[-1]: X.append(a); Y.append(b)
    return np.array(X), np.array(Y)
def table(h, pol, y, xhat, rel, first, cur, kind, st, vp):
    """The curve of half-wave h from what the engine did: under the first height the walk that was made, above it the curve
    the rebuilt values tell (per level of the bent sound their middle value)."""
    xp = bfeat.peak_of(xhat, pol)
    if kind == 'hard': c = st['c']; return Tab(c, np.array([c]), np.array([c]), xp, c)
    lb = np.append(np.arange(L0, first['os'], STEP), first['os']); q = np.interp(lb, first['o'], first['q']); xn = vp * 10 ** ((lb + q) / 20); yn = vp * 10 ** (lb / 20)
    if cur is not None: xn = np.concatenate([xn, cur[1]]); yn = np.concatenate([yn, cur[0]])
    zone = ~rel & (pol * y > 0)
    if zone.sum() >= LEAST: ym, xm = curve_from(pol * y[zone], pol * xhat[zone], float(yn[-1]), float(xn[-1])); xn = np.concatenate([xn, xm[1:]]); yn = np.concatenate([yn, ym[1:]])
    xn, yn = rising(xn, yn); return Tab(xp, xn, yn, xp, vp)
# ---------------------------------------------------------------- the engine
def analyse(y, S, vp, kept=True, sr=48000, opt=None):
    """What the declipper makes of a sound. y (n, ch), S and vp: the meter's reading and the half-waves' peaks, kept: whether
    the meter has found a bend at all (a soft flat top is only looked for then; a plain cut always).
    -> did (False: nothing of this engine, the bend engine as it is), kinds, cut (the share of all values that sit at a plain cut), xhat (the sound before the bend, in the scale
    where small values pass as they are; its 'none' half-wave walked back the careful way), tabs (per half-wave its curve, or
    None for 'none'), rebuilt (whether), test (the self-test's score), ratio (second pass against first), share (unknown)."""
    o = dict(OPT); o.update(opt or {}); N = window(sr); st = stats(y, vp); kinds = [regime(st[h], kept) for h in (0, 1)]; res = dict(did=False, kinds=kinds, st=st, test=None, ratio=None, rebuilt=False, share=0.0, second=False, cut=0.0)
    if kinds[0] == 'none' and kinds[1] == 'none': return res
    x, rel, lo, hi, first, kinds = setup(y, S, vp, kinds, st); res['kinds'] = kinds; res['share'] = float(1.0 - rel.mean()); res['cut'] = sum(st[h]['tn'] for h in (0, 1) if kinds[h] == 'hard') / float(y.size)
    if rel.all(): return res
    soft = 'soft' in kinds
    if o['test']:
        res['test'], res['d0'] = selftest(x, rel, lo, hi, o['bud_first'] if soft else o['bud_hard'], sr, N, o['core'])
        if res['test'] < TEST_MIN:                                                   # rebuilding would do harm: a plain cut stays as it is, a soft top goes the careful way
            kinds = ['none' if k == 'soft' else k for k in kinds]; res['kinds'] = kinds
            if 'hard' not in kinds: return res
            x, rel, lo, hi, first, kinds = setup(y, S, vp, kinds, st); xhat = np.where(rel, x, y); res.update(did=True, xhat=xhat, tabs=[Tab(st[h]['c'], np.array([st[h]['c']]), np.array([st[h]['c']]), st[h]['c'] * 10 ** (bengine.tidy(S[h], bengine.SPREAD)[-1] / 20), st[h]['c']) if kinds[h] == 'hard' else None for h in (0, 1)]); return res
    bud = o['bud_first'] if soft else o['bud_hard']; only, seen, off, whole = choose(rel, N // 2, o['core'], o['first']) if soft and o['first'] else (None, None, None, True)
    if whole: only = seen = off = None
    out = rebuild(x, rel, lo, hi, bud, N, o['core'], only); cur = [None, None]; relf = rel; res['whole'] = whole; res['first'] = out
    if soft:                                                                         # a long sound: the first pass on a part of it; the curve it tells serves the whole sound
        x1, rel1, lo1, hi1, cur1, did = second(y, x, rel, lo, hi, out, kinds, st, seen)
        if did:
            out1 = rebuild(x1, rel1, lo1, hi1, o['bud_second'], N, o['core']) if not rel1.all() else x1; res['ratio'] = dense(out1, N, off) / max(dense(out, N, off), 1e-30); res['out1'] = out1
            if not o['dense'] or res['ratio'] <= DENSE_MAX: out = out1; cur = cur1; relf = rel1; res['second'] = True
        if not res['second'] and not whole:                                          # the first pass is the answer after all: the rest of it
            out = rebuild(x, rel, lo, hi, bud, N, o['core'], [set(range(len(pieces(~rel[:, c], N // 2, o['core'])))) - only[c] for c in range(rel.shape[1])], out)
    res.update(did=True, rebuilt=True, xhat=out, tabs=[table(h, pol, y, out, relf, first[h], cur[h], kinds[h], st[h], vp[h]) if kinds[h] != 'none' else None for h, pol in ((0, 1.0), (1, -1.0))]); return res
def verdict(logit, top, A):
    """What the app shows: found (at all), shown (at the Sureness slider's first place), sure, kind. A plain cut whose peaks
    were rebuilt by TOP_HARD dB or more shows by itself, whatever the meter says."""
    found = bengine.kept(logit, top); shown = bengine.shown(logit, top); sure = float(bengine.sure(logit, top)); hard = False
    if A['did']:
        for h in (0, 1):
            if A['kinds'][h] == 'hard' and A['rebuilt'] and A['tabs'][h].top >= TOP_HARD: hard = True
    if hard: found = shown = True; sure = max(sure, SURE_HARD)
    tops = [A['tabs'][h].top for h in (0, 1) if A['did'] and A['tabs'][h] is not None] + ([top] if not A['did'] or None in A['tabs'] else [])
    return dict(found=found, shown=shown, sure=sure, kind='distortion' if (hard or max(tops) >= bengine.HARD_AT) else 'saturation', top=float(max(tops)))
def parts(y, S, vp, A, s):
    """Per half-wave for the fader at s: the value before the bend (positive), the curve, its top."""
    out = []
    for h, pol in ((0, 1.0), (1, -1.0)):
        if A['tabs'][h] is None: cv = bengine.Curve(bengine.careful(S[h]) if s < 1.0 else S[h], vp[h], bengine.SPREAD); out.append((cv.back(np.maximum(pol * y, 0.0)), cv))
        else: out.append((np.maximum(pol * A['xhat'], 0.0), A['tabs'][h]))
    return out
def own(y, S, vp, A, s):
    """The sound itself: s = 0 the bend taken out (flat tops rebuilt), 1 untouched, 2 driven as far again. The peak stays."""
    if not A['did']: return bengine.own(y, S, vp, s)
    if s == 1.0: return y
    (xa, ca), (xb, cb) = parts(y, S, vp, A, s); g = bengine.gamma(s, max(ca.top, cb.top)); pk = bfeat.peak_of(y); x = np.where(y >= 0, xa, -xb)
    if g <= 0.0: out = x
    else:                                                                            # what the curve does not explain of the sound in hand comes back as the fader nears its middle
        w = min(g, 1.0) ** 2; out = np.where(y >= 0, ca.fwd(g * xa) + w * (np.maximum(y, 0.0) - ca.fwd(xa)), -(cb.fwd(g * xb) + w * (np.maximum(-y, 0.0) - cb.fwd(xb))))
    return out * (pk / max(bfeat.peak_of(out), 1e-30))
def onto(b, S, vp, A, s):
    """Another sound through the found bend. Without a plain cut: as the bend engine does it (its peak driven as far into
    the meter's curve as the first sound's was), also where a soft top was rebuilt: the rebuilt peaks' height is the least
    sure thing about them. With a plain cut: driven so far that as many of its values are cut as of the first sound's."""
    if not A['did'] or 'hard' not in A['kinds']: return bengine.onto(b, S, vp, s)
    if s <= 0.0: return b
    cv = [A['tabs'][h] if A['kinds'][h] == 'hard' else bengine.Curve(S[h], vp[h], bengine.SPREAD) for h in (0, 1)]; pb = bfeat.peak_of(b); g = bengine.gamma(s, max(cv[0].top, cv[1].top)); r = np.zeros(b.size)
    for h, pol in ((0, 1.0), (1, -1.0)):
        if A['kinds'][h] == 'hard': r = np.maximum(r, pol * b.ravel() / A['st'][h]['c'])                   # how far each value reaches towards the cut
    m = int(np.floor(A['cut'] * b.size + 0.5)); top = float(np.partition(r, b.size - m)[b.size - m]) if m >= 1 else float(r.max()); k = g / max(top, 1e-30)
    out = np.where(b >= 0, cv[0].fwd(np.maximum(k * b, 0.0)), -cv[1].fwd(np.maximum(-k * b, 0.0))); return out * (pb / max(bfeat.peak_of(out), 1e-30))
