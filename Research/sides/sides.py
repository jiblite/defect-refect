# The stereo engine (reference for Source/Core/Stereo.h): HAAS (one side late) and WIDTH (a middle made wide).
#  (A) "the pair": one sound x reaches the two sides through two short, sparse filters: left = a * x, right = b * x, so
#      left * b = right * a (the cross relation). The taps are looked for with a penalty on their number, over the whole
#      sound; then fitted exactly on the samples where every tap is inside the file, in pieces of 5 ms of which those that
#      do not obey the pair (a click, a plugin's glitch) are left out; the taps that explain next to nothing go.
#  (B) "the copy": the middle kept and a filtered copy of it put at the sides: side = h * middle (taps -5 .. +5 ms).
# Both are then checked on the sound itself: x (or the copy) is worked out, and what the two sides hold beyond it is
# measured piece by piece; that says where along the sound the effect is and how much of the sound it explains.
import numpy as np
SR = 48000
K = 1920              # (A): taps looked for at 0 .. 40 ms
ORDER = 32            # (B): the whitening filter (linear prediction on the middle)
WORD = 1920           # (A): the whitening filter both sides share (linear prediction on their average spectrum, 40 ms long)
BLK = 240             # pieces of 5 ms
MOST = 56             # (A): unknowns in the exact fit, at most
NEAR = 24             # (A): a side's own past may only be used from 0.5 ms back (see lasso)
PIECES_MOST = 1800    # (A): pieces in the exact fit, at most (9 s; a longer sound: evenly spread)
KB = 240              # (B): taps of h at -5 .. +5 ms
NF = 1 << 17          # transform size of the block work
CB = NF // 2 - 2 * K  # samples per block of the correlations
LONG = 60 * SR        # (A): a longer sound: the pair is looked for in the 60 s with the most energy, then checked on all of it
LEAST = K + WORD + 20 * BLK   # a sound under 180 ms: nothing is looked for

# ---------------------------------------------------------------- small pieces
def levinson(r, order, noise):
    """Levinson's recursion on the autocorrelation r[0 .. order], with a little white noise (so the filter stays tame
    where the sound has nothing): w with w[0] = 1; a sound with that autocorrelation comes out flat."""
    a = np.zeros(order + 1); a[0] = 1.0
    if not r[0] > 0: return a
    r = np.array(r[:order + 1], float); r[0] *= 1.0 + noise; e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + (float(a[1:i] @ r[i - 1:0:-1]) if i > 1 else 0.0); k = -acc / e
        if i > 1: a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k; e *= 1.0 - k * k
        if not e > 0: break
    return a
def lpc(x, order=ORDER):
    n = len(x); return levinson(np.array([float(x[:n - k] @ x[k:]) for k in range(order + 1)]), order, 1e-6)
def fir(x, w): return np.convolve(x, w)[:len(x)]
def fir_fft(x, w):
    """x through the taps w (lags 0 ..), block by block."""
    n = len(x); B = NF - len(w) + 1; W = np.fft.rfft(w, NF); out = np.zeros(n + NF)
    for t0 in range(0, n, B): out[t0:t0 + NF] += np.fft.irfft(np.fft.rfft(x[t0:t0 + B], NF) * W, NF)
    return out[:n]
def whiten(y):
    """Both sides at the same level and through the same long filter, the one that flattens their average spectrum over
    40 ms: the cross relation holds as before, every part of the spectrum counts, and neither side can be told from
    its own past any more (a steady note would otherwise explain itself, one period back, as well as the other side
    explains it). Returns the two sides and the levels taken off."""
    s = np.sqrt(np.maximum((y ** 2).sum(0), 1e-30)); z = y / s; rLL, rRR, rLR = correlations(z); w = levinson(rLL[K:K + WORD + 1] + rRR[K:K + WORD + 1], WORD, 1e-4)
    return np.stack([fir_fft(z[:, 0], w), fir_fft(z[:, 1], w)], 1), s
def correlations(z):
    """rLL, rRR, rLR at lags -K .. K (index K = lag 0), r_XY[k] = sum_t X(t) Y(t + k), exact, block by block."""
    n = len(z); SLL = np.zeros(NF // 2 + 1, complex); SRR = SLL.copy(); SLR = SLL.copy(); pad = np.concatenate([np.zeros((K, 2)), z, np.zeros((K + CB, 2))])
    for t0 in range(0, n, CB):
        blk = pad[K + t0:K + t0 + CB]; ext = pad[t0:t0 + CB + 2 * K]        # ext starts K samples before the block
        FL = np.fft.rfft(blk[:, 0], NF); FR = np.fft.rfft(blk[:, 1], NF); EL = np.fft.rfft(ext[:, 0], NF); ER = np.fft.rfft(ext[:, 1], NF)
        SLL += np.conj(FL) * EL; SRR += np.conj(FR) * ER; SLR += np.conj(FL) * ER
    return tuple(np.fft.irfft(S, NF)[:2 * K + 1] for S in (SLL, SRR, SLR))

# ---------------------------------------------------------------- (A) the pair
class Gram:
    """R(t) = sum_j b_j L(t - j) - sum_{j >= 1} a_j R(t - j)  (a_0 = 1), with sums over the whole sound (fast, a little
    off at the file's two ends). Unknown i <= K: b_i; unknown K + j: a_j."""
    def __init__(self, rLL, rRR, rLR):
        self.rLL, self.rRR, self.rLR = rLL, rRR, rLR; self.P = 2 * K + 1; self.xy = np.concatenate([rLR[K:2 * K + 1], -rRR[K + 1:2 * K + 1]]); self.yy = rRR[K]
        self.d = np.concatenate([np.full(K + 1, rLL[K]), np.full(K, rRR[K])])
    def col(self, i):
        if i <= K: return np.concatenate([self.rLL[K - i:2 * K + 1 - i], -self.rLR[K + i - 1:i - 1:-1] if i >= 1 else -self.rLR[K - 1::-1][:K]])
        j = i - K; return np.concatenate([-self.rLR[K - j:2 * K + 1 - j], self.rRR[K + 1 - j:2 * K + 1 - j]])
def lasso(G, lam, sweeps=40, tol=1e-5):
    """Coordinate descent on 1/2 |R - X w|^2 + lam |w|_1 through the correlations. The predicted side's own last half
    millisecond is not used: a pair of filters that differ only there is a difference in tone between the sides, not
    one of the effects looked for, and whatever colour the whitening has left in the sound would let a side be
    explained from its own last few samples instead of from the other side."""
    w = np.zeros(G.P); grad = -G.xy.copy(); d = G.d; free = np.ones(G.P, bool); free[K + 1:K + NEAR] = False
    for sweep in range(sweeps):
        cand = np.where(free & ((w != 0) | (np.abs(grad) > lam)))[0]
        if len(cand) == 0: break
        order = cand[np.argsort(-np.abs(grad[cand]), kind='stable')][:300]; moved = 0.0
        for rep in range(6):
            mv = 0.0
            for i in order:
                z = w[i] - grad[i] / d[i]; nw = np.sign(z) * max(abs(z) - lam / d[i], 0.0); dw = nw - w[i]
                if dw != 0.0: grad += dw * G.col(i); w[i] = nw; mv = max(mv, abs(dw))
            moved = max(moved, mv)
            if mv < tol: break
        if moved < tol and sweep > 0: break
    return w
def block_sums(zL, zR, sup):
    """The regression on the samples K + WORD .. n (every tap inside the file, the whitening filter filled), summed per
    piece of 5 ms: X^T X, X^T y, y^T y. A long sound: PIECES_MOST pieces, evenly spread."""
    n = len(zL); t0 = K + WORD; nb = max(1, (n - t0) // BLK); pick = np.arange(nb) if nb <= PIECES_MOST else (np.arange(PIECES_MOST) * nb) // PIECES_MOST
    S = len(sup); Ab = np.zeros((len(pick), S, S)); bb = np.zeros((len(pick), S)); yb = np.zeros(len(pick))
    for q, k in enumerate(pick):
        a = t0 + k * BLK; e = min(a + BLK, n); X = np.stack([zL[a - i:e - i] if i <= K else -zR[a - (i - K):e - (i - K)] for i in sup], 1); y = zR[a:e]
        Ab[q] = X.T @ X; bb[q] = X.T @ y; yb[q] = float(y @ y)
    return Ab, bb, yb
def solve(A, bvec, yy):
    """Least squares through the Cholesky factor (a little ridge; more if the matrix does not factor): the inverse, the taps, the rest."""
    n = len(A); ridge = 1e-10 * max(float(np.trace(A)), 1e-30) / max(n, 1)
    for _ in range(8):
        try: L = np.linalg.cholesky(A + ridge * np.eye(n)); break
        except np.linalg.LinAlgError: ridge *= 100.0
    else: return np.zeros((n, n)), np.zeros(n), float(yy)
    Li = np.linalg.inv(L); Ai = Li.T @ Li; w = Ai @ bvec; return Ai, w, float(yy - bvec @ w)
def prune(A, bvec, yy, n, slack=10 ** (-40 / 10), grow=0.10):
    """Backward elimination: the tap whose loss costs least goes while the unexplained rest stays under (1 + grow) * best + slack."""
    idx = np.arange(n)
    if n == 0: return idx
    Ai, w, res = solve(A, bvec, yy); limit = (1 + grow) * max(res, 0.0) + slack * yy
    while len(idx) > 0:
        cost = w ** 2 / np.maximum(np.diag(Ai), 1e-30); j = int(np.argmin(cost))
        if res + cost[j] > limit: break
        idx = np.delete(idx, j)
        if len(idx) == 0: break
        Ai, w, res = solve(A[np.ix_(idx, idx)], bvec[idx], yy)
    return idx
def robust(Ab, bb, yb, idx, live, rounds=10, drop=0.05, need=100.0):
    """Trimmed fit: the pieces that fit worst are left out, 5 % at a time (at most half). The trimmed fit is taken only if
    it explains the pieces still in at least `need` times better than the fit over everything explained them all: then
    most of the sound obeys the pair exactly and the rest is something else (a click, a glitch of the plugin). In that
    case every piece that fits about as well as the middle one is in, whatever was left out on the way.
    Returns keep (bool per piece), the taps' values, the unexplained share."""
    As = Ab[:, idx][:, :, idx]; bs = bb[:, idx]
    def fit(k):
        yy = float(yb[k].sum()); Ai, w, res = solve(As[k].sum(0), bs[k].sum(0), yy); return w, max(res, 0.0) / max(yy, 1e-30)
    def errors(w): return (yb - 2 * bs @ w + np.einsum('i,kij,j->k', w, As, w)) / np.maximum(yb, 1e-9 * yb.max())
    w0, res0 = fit(live); k, w, res = live, w0, res0
    for _ in range(rounds):
        rel = errors(w); rel[~k] = -1.0; nd = max(1, int(drop * int(live.sum()))); k2 = k.copy(); k2[np.argsort(-rel, kind='stable')[:nd]] = False
        if not k2.any(): break
        k = k2; w, res = fit(k)
        if res < 1e-9: break
    if not res * need < res0: return live, w0, res0
    for _ in range(2):
        rel = errors(w); k2 = live & (rel <= max(100.0 * float(np.median(rel[k])), 1e-4))
        if not k2.any(): break
        k = k2; w, res = fit(k)
    return k, w, res
def one_way(zL, zR, cor, mu=0.01):
    rLL, rRR, rLR = cor; G = Gram(rLL, rRR, rLR); w = lasso(G, mu * float(np.abs(G.xy).max())); sup = np.where(w != 0)[0]
    if len(sup) > MOST // 2: sup = sup[np.argsort(-np.abs(w[sup]), kind='stable')[:MOST // 2]]
    near = set(int(i) for i in sup) | {0}                 # the exact fit may also use the taps next to the found ones and the direct tap of the predicting side
    for i in sup:
        for d in (-1, 1):
            j = int(i) + d
            if (i <= K and 0 <= j <= K) or (i > K and K + NEAR <= j <= 2 * K): near.add(j)
    sup = np.array(sorted(near), int)
    if len(sup) > MOST: sup = np.array(sorted(sorted(near, key=lambda i: (-abs(w[i]), i))[:MOST]), int)
    Ab, bb, yb = block_sums(zL, zR, sup); idx = np.arange(len(sup)); live = yb > 1e-9 * yb.max()
    keep, wr, res = robust(Ab, bb, yb, idx, live)
    for _ in range(2):                                   # taps that explain next to nothing go; then the pieces are chosen again with the taps left
        sub = prune(Ab[keep][:, idx][:, :, idx].sum(0), bb[keep][:, idx].sum(0), float(yb[keep].sum()), len(idx)); idx = idx[sub]
        keep, wr, res = robust(Ab, bb, yb, idx, live)
    b = np.zeros(K + 1); a = np.zeros(K + 1); a[0] = 1.0
    for i, v in zip(sup[idx], wr):
        if i <= K: b[i] = v
        else: a[i - K] = v
    return a, b, res, len(idx), float(yb[keep].sum() / max(yb.sum(), 1e-30))
def loudest(y, most=LONG, hop=SR // 2):
    """The stretch of `most` samples with the most energy (looked for in steps of half a second): start, length."""
    n = len(y)
    if n <= most: return 0, n
    nb = n // hop; k = most // hop; c = np.concatenate([[0.0], np.cumsum((y[:nb * hop] ** 2).reshape(nb, hop, 2).sum((1, 2)))]); return int(np.argmax(c[k:nb + 1] - c[:nb + 1 - k])) * hop, most
def pair(y, mu=0.01):
    """(A): dict(a, b, taps, way, res, fits, check) or None. a, b: the two filters (K + 1 taps each, the largest tap of
    all = 1). The fit is made both ways round (the left side told from the right one and its own past, and the other
    way), and the one that explains more of the sound itself (check_pair) is taken.
    res: the share of the predicted side left unexplained in the pieces that fit; fits: the share of the sound in them."""
    whole = y; t0, n0 = loudest(y); y = y[t0:t0 + n0]
    eL, eR = float((y[:, 0] ** 2).sum()), float((y[:, 1] ** 2).sum())
    if len(y) < LEAST or min(eL, eR) < 1e-8 * max(eL, eR, 1e-30): return None
    z, s = whiten(y); rLL, rRR, rLR = correlations(z); best = None
    for way in 'ab':
        if way == 'a': a, b, res, n, fits = one_way(z[:, 0], z[:, 1], (rLL, rRR, rLR), mu)
        else: b, a, res, n, fits = one_way(z[:, 1], z[:, 0], (rRR, rLL, rLR[::-1]), mu)
        a = a * s[0]; b = b * s[1]; k = max(float(np.abs(a).max()), float(np.abs(b).max()))
        if not (np.abs(a).max() > 0 and np.abs(b).max() > 0): continue          # one side told from its own past alone: no pair
        a = a / k; b = b / k; ck = check_pair(y, a, b); cost = (1.0 - ck['covered']) + ck['inside'] + 0.002 * (np.count_nonzero(a) + np.count_nonzero(b))
        if best is None or cost < best['cost']: best = dict(a=a, b=b, res=res, taps=n, way=way, fits=fits, check=ck, cost=cost)
    if best is not None and len(whole) > len(y): best['check'] = check_pair(whole, best['a'], best['b'])
    return best

# ---------------------------------------------------------------- block work: one filter pair on the whole sound
def _blocks(n, seg=1 << 15):
    hop = seg // 2; t0 = -hop
    while t0 < n: yield t0; t0 += hop
def filtered(sigs, H, n_out=None, seg=1 << 15):
    """sum_i sigs[i] through H[i] (transfer functions on NF // 2 + 1 points, any phase, responses well under a second),
    by overlapping windows: exact up to the responses' far ends."""
    n = len(sigs[0]); n_out = n_out or n; win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(seg) / seg); off = (NF - seg) // 2; out = np.zeros(n_out + NF)
    for t0 in _blocks(n, seg):
        a = max(t0, 0); e = min(t0 + seg, n)
        if e <= a: continue
        X = 0
        for v, h in zip(sigs, H):
            buf = np.zeros(NF); buf[off + a - t0:off + e - t0] = v[a:e] * win[a - t0:e - t0]; X = X + np.fft.rfft(buf) * h
        x = np.fft.irfft(X, NF); p = t0 - off            # x[i] belongs to time p + i
        i0 = max(0, -p); i1 = min(NF, n_out - p)
        if i1 > i0: out[p + i0:p + i1] += x[i0:i1]
    return out[:n_out]
def sparse_conv(x, v):
    """x through the taps v (lags 0 ..)."""
    out = np.zeros(len(x))
    for i in np.where(v != 0)[0]: out[i:] += v[i] * x[:len(x) - i]
    return out
def the_one_sound(y, a, b, eps=1e-4):
    """The one sound behind both sides: X = (A* L + B* R) / (|A|^2 + |B|^2). With left = a * x, right = b * x it is x itself."""
    A = np.fft.rfft(a, NF); B = np.fft.rfft(b, NF); D = np.abs(A) ** 2 + np.abs(B) ** 2; W = 1.0 / (D + eps * float(D.max()))
    return filtered([y[:, 0], y[:, 1]], [np.conj(A) * W, np.conj(B) * W])
def pieces(p, u, thr, close=50, least=4):
    """Per piece of 5 ms: p its power, u what is unexplained of it. A piece obeys when u <= thr * p. Spans: runs of such
    pieces, holes of up to `close` pieces (quiet or not obeying) closed, runs under `least` pieces dropped.
    Returns the mask (bool per piece), the spans [(first piece, one past the last)], the share of the sound's power in obeying pieces,
    the unexplained share inside them."""
    live = p > 1e-7 * float(p.max()) if len(p) and p.max() > 0 else np.zeros(len(p), bool); ok = live & (u <= thr * p)
    covered = float(p[ok].sum() / max(float(p.sum()), 1e-30)); inside = float(u[ok].sum() / max(float(p[ok].sum()), 1e-30)) if ok.any() else 1.0
    spans = []; i = 0; nb = len(ok)
    while i < nb:
        if not ok[i]: i += 1; continue
        j = i
        while True:
            while j < nb and ok[j]: j += 1
            g = j
            while g < nb and not ok[g] and g - j <= close: g += 1
            if g < nb and ok[g] and g - j <= close: j = g
            else: break
        if j - i >= least: spans.append((i, j))
        i = j
    return ok, spans, covered, inside
def check_pair(y, a, b, thr=10 ** (-12 / 10)):
    """What the pair explains of the sound itself: x, the two sides' rests, and per piece how much is unexplained.
    beyond: what is unexplained in the obeying pieces against what one sound at two levels (no filters at all: a
    sound panned somewhere) leaves unexplained there, dB. A sound with little at its sides is explained well by any
    pair close to that; a pair that is really there explains far more than the panned sound does."""
    x = the_one_sound(y, a, b); r = np.stack([y[:, 0] - sparse_conv(x, a), y[:, 1] - sparse_conv(x, b)], 1); nb = len(y) // BLK
    if nb == 0: return dict(x=x, rest=r, ok=np.zeros(0, bool), spans=[], covered=0.0, inside=1.0, beyond=0.0)
    p = (y[:nb * BLK] ** 2).reshape(nb, BLK, 2).sum((1, 2)); u = (r[:nb * BLK] ** 2).reshape(nb, BLK, 2).sum((1, 2)); ok, spans, covered, inside = pieces(p, u, thr)
    beyond = 0.0
    if ok.any():
        m = np.repeat(ok, BLK); L = y[:nb * BLK, 0][m]; R = y[:nb * BLK, 1][m]; ll = float(L @ L); rr = float(R @ R); lr = float(L @ R)
        plain = 0.5 * (ll + rr) - np.sqrt((0.5 * (ll - rr)) ** 2 + lr ** 2)          # the smaller eigenvalue: what the best panned sound leaves
        beyond = float(10 * np.log10(max(float(u[ok].sum()), 1e-30) / max(plain, 1e-30))) if plain > 1e-12 * (ll + rr) else 0.0
    return dict(x=x, rest=r, ok=ok, spans=spans, covered=covered, inside=inside, beyond=beyond)
def shape(a, b):
    """Numbers of the pair: how many taps, how far apart the two sides' main taps are (samples), how far its taps reach,
    how much side it makes of white noise with both sides brought to one level (dB under the middle)."""
    ia = int(np.argmax(np.abs(a))); ib = int(np.argmax(np.abs(b))); big = np.where((np.abs(a) >= 0.1 * np.abs(a).max()) | (np.abs(b) >= 0.1 * np.abs(b).max()))[0]
    an = a / np.sqrt((a ** 2).sum()); bn = b / np.sqrt((b ** 2).sum()); side = float(((an - bn) ** 2).sum()); mid = float(((an + bn) ** 2).sum())
    return dict(taps=int(np.count_nonzero(a) + np.count_nonzero(b)), apart=abs(ia - ib), reach=int(big.max() - big.min()) if len(big) else 0, made=10 * np.log10(max(side, 1e-12) / max(mid, 1e-12)))

# ---------------------------------------------------------------- (B) the side as a filtered copy of the middle
def levinson_rhs(col, rhs):
    """Solve T h = rhs for the symmetric Toeplitz matrix T with first column col (Levinson's recursion)."""
    n = len(rhs); f = np.zeros(n); x = np.zeros(n); f[0] = 1.0 / col[0]; x[0] = rhs[0] / col[0]
    for i in range(1, n):
        ef = float(col[i:0:-1] @ f[:i]); den = 1.0 - ef * ef
        if not abs(den) > 1e-300: break
        fn = np.zeros(i + 1); fn[:i] = f[:i] / den; fn[1:] -= ef / den * f[i - 1::-1][:i]      # the forward vector of size i + 1; the backward one is its mirror
        f[:i + 1] = fn; ex = float(col[i:0:-1] @ x[:i]); x[:i + 1] += (rhs[i] - ex) * fn[::-1]
    return x
def side_copy(y):
    """h (2 KB + 1 taps, index KB = lag 0) with side ~ h * middle, fitted on whitened signals over the whole sound.
    share: the part of the side's power that is such a copy, without what lag 0 alone explains (that is balance)."""
    m = 0.5 * (y[:, 0] + y[:, 1]); s = 0.5 * (y[:, 0] - y[:, 1]); em, es = float(m @ m), float(s @ s)
    if len(y) < LEAST or em < 1e-12 or es < 1e-10 * em: return None
    w = lpc(m); z = np.stack([fir(m, w), fir(s, w)], 1); cm, cs, cms = correlations_kb(z)
    P = 2 * KB + 1; col = cm[KB:KB + P].copy(); col[0] *= 1.0 + 1e-4; rhs = cms            # <m(t - j), s(t)> for j = -KB .. KB is r_ms[j]
    h = levinson_rhs(col, rhs); yy = float(cs); expl = float(rhs @ h) / max(yy, 1e-30); expl0 = float(rhs[KB] ** 2 / col[0]) / max(yy, 1e-30)
    return dict(h=h, share=max(0.0, expl - expl0), all=expl, balance=float(h[KB]), side_db=10 * np.log10(es / em))
def correlations_kb(z):
    """Of the whitened middle (z[:, 0]) and side (z[:, 1]): r_mm at lags -2 KB .. 2 KB (index 2 KB = lag 0... returned from lag -KB), the side's power, r_ms at lags -KB .. KB."""
    rLL, rRR, rLR = correlations(z); return rLL[K - KB:], float(rRR[K]), rLR[K - KB:K + KB + 1]
def conv_centered(m, h):
    """h * m with h's index KB at lag 0, same length as m."""
    H = np.fft.rfft(np.concatenate([h[KB:], np.zeros(NF - len(h)), h[:KB]]), NF); return filtered([m], [H])
def check_copy(y, h, thr=10 ** (-10 / 10)):
    m = 0.5 * (y[:, 0] + y[:, 1]); s = 0.5 * (y[:, 0] - y[:, 1]); c = conv_centered(m, h); r = s - c; nb = len(y) // BLK
    if nb == 0: return dict(copy=c, ok=np.zeros(0, bool), spans=[], covered=0.0, inside=1.0)
    p = (s[:nb * BLK] ** 2).reshape(nb, BLK).sum(1); u = (r[:nb * BLK] ** 2).reshape(nb, BLK).sum(1); ok, spans, covered, inside = pieces(p, u, thr)
    return dict(copy=c, ok=ok, spans=spans, covered=covered, inside=inside)

# ---------------------------------------------------------------- the verdict
APART = 60            # HAAS: the two sides' main taps at least 1.25 ms apart
def find(y, rule=None):
    """What is on the sound: dict(kind 'haas' | 'width' | None, mode 'pair' | 'copy', ...). The numbers behind the verdict are in it either way."""
    R = dict(taps=8, reach=60, made=-20.0, covered=0.5, inside=-20.0, beyond=-10.0, share=0.9, side=-15.0, copy_covered=0.5); R.update(rule or {})
    f = dict(kind=None, mode=None, n=len(y)); p = pair(y)
    if p is not None:
        sh = shape(p['a'], p['b']); ck = p['check']; f.update(pair=p, shape=sh, pair_check=ck)
        if sh['taps'] <= R['taps'] and sh['reach'] >= R['reach'] and sh['made'] >= R['made'] and ck['covered'] >= R['covered'] and 10 * np.log10(max(ck['inside'], 1e-30)) <= R['inside'] and ck['beyond'] <= R['beyond']:
            f.update(mode='pair', kind='haas' if sh['apart'] >= APART else 'width', spans=ck['spans']); return f
    g = side_copy(y)
    if g is not None:
        f.update(copy=g)
        if g['share'] >= R['share'] and g['side_db'] >= R['side']:
            ck = check_copy(y, g['h']); f.update(copy_check=ck)
            if ck['covered'] >= R['copy_covered']: f.update(mode='copy', kind='width', spans=ck['spans'])
    return f

# ---------------------------------------------------------------- the Amount fader
def gate(spans, n, fade=480):
    """1 inside the spans, 0 outside; the way in and out (10 ms, raised cosine) lies outside the span, where the sound
    is quiet or something else: a span's first and last moments are worked on in full."""
    g = np.zeros(n); ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade)
    for i, j in spans:
        a = i * BLK; e = n if (j + 1) * BLK > n else j * BLK; g[a:e] = 1.0
        k = min(fade, a); g[a - k:a] = np.maximum(g[a - k:a], ramp[fade - k:])
        k = min(fade, n - e); g[e:e + k] = np.maximum(g[e:e + k], ramp[::-1][:k])
    return g
def shift(x, d, half=32):
    """x moved by d samples (later for d > 0, earlier for d < 0; any fraction): a whole shift and a windowed-sinc filter for the rest."""
    n = len(x); k = int(np.floor(d)); fr = d - k
    if fr > 1.0 - 1e-9: k += 1; fr = 0.0
    pad = abs(k) + half + 1; xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])      # xp[u] = x[u - pad]
    if fr < 1e-9: return xp[pad - k:pad - k + n].copy()
    t = np.arange(-half + 1, half + 1) - fr; h = np.sinc(t) * (0.5 + 0.5 * np.cos(np.pi * t / half)); h /= h.sum(); out = np.zeros(n)
    for j in range(2 * half): q = j - half + 1; out += h[j] * xp[pad - k - q:pad - k - q + n]      # y[i] = sum_j h[j] x[i - k - q_j]
    return out
def one_tap(v):
    """(delay, gain) if the filter is a single tap, or two neighbours that stand for a tap between two samples; else None."""
    idx = np.where(v != 0)[0]
    if len(idx) == 1: return float(idx[0]), float(v[idx[0]])
    if len(idx) == 2 and idx[1] == idx[0] + 1 and v[idx[0]] * v[idx[1]] > 0: g = float(v[idx].sum()); return float((idx * v[idx]).sum() / g), g
    return None
def pair_at(a, b, s, c):
    """The pair with Amount at s (0 .. 2; 1 = as found) against the plain sound at level c on both sides (s = 0).
    One tap on each side, both the same way up ('warp'): the late side comes nearer in time and the two levels come
    together (to the right: later, up to 40 ms, and further apart); returned per side: (the factor on the found side,
    how far it moves in time, samples). Anything else ('taps'): left of the middle a straight line between the plain
    sound and the found pair; right of it everything but each side's direct sound grows; returned: the two filters."""
    ta, tb = one_tap(a), one_tap(b)
    if ta is not None and tb is not None and ta[1] > 0 and tb[1] > 0:
        d0 = min(ta[0], tb[0]); return ('warp',) + tuple(((g / c) ** (s - 1.0), min(s * (d - d0), float(K)) - (d - d0)) for d, g in (ta, tb))
    o = []
    for v in (a, b):
        w = s * v
        if s <= 1.0: w[0] += (1.0 - s) * c
        else: w[0] = v[0]
        o.append(w)
    return ('taps',) + tuple(o)
def level_of(a, b): return float(np.sqrt(0.5 * ((a ** 2).sum() + (b ** 2).sum())))
def own(y, f, s):
    """The sound itself with the group's Amount at s: 0 = the effect out, 1 = untouched, 2 = twice."""
    if f['mode'] is None or abs(s - 1.0) < 1e-12: return y.copy()
    n = len(y); g = gate(f['spans'], n)
    if f['mode'] == 'copy':
        c = (1.0 - s) * g * f['copy_check']['copy']; return np.stack([y[:, 0] - c, y[:, 1] + c], 1)
    a, b = f['pair']['a'], f['pair']['b']; x = f['pair_check']['x']; kind, pa, pb = pair_at(a, b, s, level_of(a, b)); out = y.copy()
    for ch, (v, pv) in enumerate(((a, pa), (b, pb))):
        was = sparse_conv(x, v); out[:, ch] += g * ((pv[0] * shift(was, pv[1]) if kind == 'warp' else sparse_conv(x, pv)) - was)
    return out
def onto(z, f, s):
    """The found effect on another sound: 0 = none, 1 = as found, 2 = twice. Each side goes through its own filter (a
    sound with nothing at its sides gets exactly what the effect does; a wide one keeps what it has at its sides)."""
    if f['mode'] is None or s < 1e-12: return z.copy()
    if f['mode'] == 'copy':
        m = 0.5 * (z[:, 0] + z[:, 1]); c = s * conv_centered(m, f['copy']['h']); return np.stack([z[:, 0] + c, z[:, 1] - c], 1)
    a, b = f['pair']['a'], f['pair']['b']; c = level_of(a, b); kind, pa, pb = pair_at(a / c, b / c, s, 1.0)
    if kind == 'warp': return np.stack([pv[0] * shift(sparse_conv(z[:, ch], v / c), pv[1]) for ch, (v, pv) in enumerate(((a, pa), (b, pb)))], 1)
    return np.stack([sparse_conv(z[:, 0], pa), sparse_conv(z[:, 1], pb)], 1)
