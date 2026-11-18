# The chorus engine's prototype: the voices of a chorus (copies of the sound whose delay moves round and round, 3 to 45 ms)
# found on a sound, taken out of it and put onto another sound.
#   picture(y1)      per frame, stretch and delay: how well the frame matches a slightly stretched version of itself. A copy
#                    whose delay grows by r per second is the sound played r slower: against the frame stretched by r it is a
#                    sharp peak at its delay (in a plain autocorrelation the peak smears over the delays it passes through)
#   peaks(A, act)    the strongest peaks of every frame: delay, stretch, height
#   voices(P, T)     curves that repeat (sine or triangle) through those peaks, one per voice: the delay over time
import sys, numpy as np, scipy.signal as ss, scipy.fft as sfft, scipy.ndimage as ndi
sys.path[:0] = ["/tmp/claude-0/v2/chorus"]
import prim
SR = 48000
DEC = 2; FS = SR // DEC; W = 2048; HOP = 512                 # the picture is made at 24 kHz: frames of 85 ms every 21 ms (longer frames smear a voice that turns quickly)
FPS = FS / HOP
TMIN, TMAX = int(0.003 * FS), int(0.045 * FS)                # delays looked at: 3 to 45 ms (in samples at FS)
RSTEP = 0.001; RHO = np.arange(-0.06, 0.06 + RSTEP / 2, RSTEP)       # the copy's delay may grow or shrink by up to 6 % of the time
LEVEL = 1e-6                                                 # frequencies this far under the frame's strongest (as power) count less and less: where the sound has nothing, only noise would be compared
NPK = 8                                                      # peaks kept per frame
RATES = np.geomspace(0.08, 4.0, 1600)                        # how often a voice's delay may go round, Hz (steps of 0.25 %; the rate is fitted finely afterwards)
TOL = 1.5                                                    # samples at FS: a peak this near a curve belongs to it
RTOL = 0.004                                                 # ... and its stretch must match the curve's slope this well

def frames_of(y1):
    x = ss.resample_poly(y1, 1, DEC); T = len(x) // HOP + 1; xp = np.concatenate([np.zeros(W // 2), x, np.zeros(W + HOP)])
    return xp[np.arange(T)[:, None] * HOP + np.arange(W)[None, :]]

def warped(F, rho):
    """Every frame stretched about its middle: s(t0 + (t - t0)(1 - rho))."""
    t = np.arange(W) - W / 2.0; u = t * (1.0 - rho) + W / 2.0; i0 = np.clip(np.floor(u).astype(int), 0, W - 2); f = u - i0
    return F[:, i0] * (1 - f) + F[:, i0 + 1] * f

def level(S, act, sm=33):
    """The sound's own tilt taken off before anything is compared: one weight per frequency from the spectrum averaged over
    the whole sound (a moving copy's comb is not in that average, a steep fall from bass to treble is). Without it a dull
    sound is compared by its lowest octave alone, which says little about a delay."""
    P = (np.abs(S[act]) ** 2).mean(0) if act.any() else np.ones(S.shape[1]); P = np.maximum(ndi.uniform_filter1d(P, sm, mode='nearest'), 0.0); return 1.0 / np.sqrt(P + 1e-9 * P.max() + 1e-30)

NFFT = 4096; LREF = W + TMAX                                  # each frame is compared with a longer piece of the stretched sound, reaching TMAX further back: a copy 45 ms late is then seen as well as one 5 ms late
def picture(y1, sm=129):
    """One channel -> (A (T, len(RHO), TMAX - TMIN) float32, act (T,) frames that sound). A[t, j, i]: how well frame t matches the
    sound stretched by RHO[j] about the frame's middle and read TMIN + i samples earlier. Every frequency is divided by the
    sound's smoothed strength around it (not by its own strength: then a copy that hardly moves would not show at all)."""
    x = prim.dec2(y1); T = len(x) // HOP + 1; front = int((W // 2 + TMAX) * 1.07) + 4; xp = np.concatenate([np.zeros(front), x, np.zeros(W + HOP)]); mid = front + np.arange(T) * HOP
    F = xp[mid[:, None] + np.arange(-(W // 2), W // 2)[None, :]]; e = (F ** 2).sum(1); act = e > e.max() * 1e-4
    f = np.zeros((T, NFFT)); f[:, TMAX:TMAX + W] = F * np.hanning(W); S = sfft.rfft(f, axis=1); tilt = level(S, act); S = S * tilt; P0 = np.maximum(ndi.uniform_filter1d(np.abs(S) ** 2, sm, axis=1, mode='nearest'), 0.0)
    tap = np.ones(LREF); k = 64; tap[:k] = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k); tap[-k:] = tap[:k][::-1]; rel = np.arange(LREF) - (W // 2 + TMAX); A = np.zeros((T, len(RHO), TMAX - TMIN), np.float32); r = np.zeros((T, NFFT))
    for j, rho in enumerate(RHO):
        u = rel * (1.0 - rho); i0 = np.floor(u).astype(int); fr = u - i0; r[:, :LREF] = (xp[mid[:, None] + i0[None, :]] * (1 - fr) + xp[mid[:, None] + i0[None, :] + 1] * fr) * tap
        Sr = sfft.rfft(r, axis=1) * tilt; Pr = np.maximum(ndi.uniform_filter1d(np.abs(Sr) ** 2, sm, axis=1, mode='nearest'), 0.0); X = S * np.conj(Sr); q = np.sqrt(P0 * Pr)
        X /= q + LEVEL * q.max(1, keepdims=True) + 1e-30; A[:, j] = sfft.irfft(X, NFFT, axis=1)[:, TMIN:TMAX]
    return A, act

def ride(y1, v, sm=129):
    """How well the sound matches itself along one voice's curve, frame by frame: the picture's value at the voice's own
    delay and stretch (the best of the three nearest delays). (T,) and the frames that sound."""
    x = prim.dec2(y1); T = len(x) // HOP + 1; front = int((W // 2 + TMAX) * 1.07) + 4; xp = np.concatenate([np.zeros(front), x, np.zeros(W + HOP)]); mid = front + np.arange(T) * HOP
    F = xp[mid[:, None] + np.arange(-(W // 2), W // 2)[None, :]]; e = (F ** 2).sum(1); act = e > e.max() * 1e-4
    f = np.zeros((T, NFFT)); f[:, TMAX:TMAX + W] = F * np.hanning(W); S = sfft.rfft(f, axis=1); tilt = level(S, act); S = S * tilt; P0 = np.maximum(ndi.uniform_filter1d(np.abs(S) ** 2, sm, axis=1, mode='nearest'), 0.0)
    tap = np.ones(LREF); k = 64; tap[:k] = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k); tap[-k:] = tap[:k][::-1]; rel = np.arange(LREF) - (W // 2 + TMAX)
    d, rho = curve(v, np.arange(T)); rho = np.clip(rho, RHO[0], RHO[-1]); u = rel[None, :] * (1.0 - rho[:, None]); i0 = np.floor(u).astype(int); fr = u - i0; r = np.zeros((T, NFFT))
    r[:, :LREF] = (xp[mid[:, None] + i0] * (1 - fr) + xp[mid[:, None] + i0 + 1] * fr) * tap; Sr = sfft.rfft(r, axis=1) * tilt; Pr = np.maximum(ndi.uniform_filter1d(np.abs(Sr) ** 2, sm, axis=1, mode='nearest'), 0.0)
    X = S * np.conj(Sr); q = np.sqrt(P0 * Pr); X /= q + LEVEL * q.max(1, keepdims=True) + 1e-30; A = sfft.irfft(X, NFFT, axis=1)
    lag = np.clip(np.round(d / (1.0 - rho)).astype(int), 1, NFFT // 2 - 2); t = np.arange(T); return np.maximum(np.maximum(A[t, lag - 1], A[t, lag]), A[t, lag + 1]), act

def peaks(A, act, k=NPK, floor=None):
    """The k strongest peaks of each sounding frame: rows (frame, the copy's delay at the frame's middle in samples at FS, stretch, height). A peak is a cell
    that tops its neighbours in delay and stretch; delay and stretch are read to a fraction of a cell."""
    T, R, L = A.shape; out = []; floor = 4.0 * float(np.median(np.median(np.abs(A[act]).reshape(int(act.sum()), -1), axis=1))) if floor is None and act.any() else (0.0 if floor is None else floor)        # four times the usual size of a cell: the middle one of the frames' own middle values
    mx = ndi.maximum_filter(A, size=(1, 9, 9), mode='nearest'); is_pk = (A == mx) & (A > floor)
    for t in np.where(act)[0]:
        j, i = np.where(is_pk[t]); h = A[t, j, i]
        if len(h) == 0: continue
        o = np.argsort(-h)[:k]
        for jj, ii, hh in zip(j[o], i[o], h[o]):
            di = 0.0; dj = 0.0
            if 0 < ii < L - 1: a, b, c = A[t, jj, ii - 1], hh, A[t, jj, ii + 1]; d = a - 2 * b + c; di = 0.5 * (a - c) / d if d < 0 else 0.0
            if 0 < jj < R - 1: a, b, c = A[t, jj - 1, ii], hh, A[t, jj + 1, ii]; d = a - 2 * b + c; dj = 0.5 * (a - c) / d if d < 0 else 0.0
            rho = RHO[jj] + dj * RSTEP; out.append((t, (ii + di + TMIN) * (1.0 - rho), rho, hh))        # against the frame stretched by rho the copy sits at delay / (1 - rho)
    return np.array(out).reshape(-1, 4)

def shape(ph, kind):
    """One round of a voice's movement, -1 .. 1: a sine (0) or a triangle (1), and its slope per turn."""
    if kind == 0: return np.sin(2 * np.pi * ph), 2 * np.pi * np.cos(2 * np.pi * ph)
    p = ph - np.floor(ph); return 1.0 - 4.0 * np.abs(p - 0.5), np.where(p < 0.5, 4.0, -4.0)

def curve(v, t):
    """A voice's delay (samples at FS) and its stretch at the frames t."""
    s, ds = shape(v['rate'] * t / FPS + v['phase'], v['kind']); return v['mid'] + v['half'] * s, v['half'] * ds * v['rate'] / FS

def near(P, v, k=1.0):
    """Which peaks lie on the voice's curve (delay within k * TOL, stretch matching its slope)."""
    d, r = curve(v, P[:, 0]); return (np.abs(P[:, 1] - d) < k * TOL) & (np.abs(P[:, 2] - r) < k * RTOL)

def rounds(P, T, most=8, K=16):
    """How often the peaks' delays come round: the rates (Hz) at which one round looks like the others. Each peak is first
    moved along its own stretch to the middle of its place in the round (so a quick voice does not smear), then counted per
    round, place in the round and delay. The score is how alike the rounds are (their counts multiplied round against
    round, 1 = all alike): what stands still (the sound's own pitch) is taken off first, and what happens once does not
    count at any rate. Strongest first: (rate, score)."""
    t = P[:, 0] / FPS; h = P[:, 3]; lo = TMIN * 0.5; nd = int((TMAX * 1.1 - lo) / 2.0) + 3; sc = np.zeros(len(RATES))
    for i, r in enumerate(RATES):
        n = int(np.ceil(r * T / FPS))
        if r * T / FPS < 1.5: continue                                     # the curve must go round one and a half times within the sound
        x = r * t; k = np.minimum(x.astype(int), n - 1); ph = x - np.floor(x); b = np.minimum((ph * K).astype(int), K - 1); dp = P[:, 1] + P[:, 2] * ((b + 0.5) / K - ph) / r * FS
        cell = np.clip(((dp - lo) / 2.0).astype(int), 0, nd - 1); H = np.bincount((k * K + b) * nd + cell, weights=h, minlength=n * K * nd).reshape(n, K, nd); H = H[:, :, :-1] + H[:, :, 1:]
        H = H - H.mean(1, keepdims=True); S = H.sum(0); own = (H ** 2).sum()
        if own > 0: sc[i] = ((S ** 2).sum() - own) / ((n - 1) * own)
    out = []; s2 = sc.copy()
    for _ in range(most):
        i = int(np.argmax(s2))
        if s2[i] <= 0: break
        out.append((float(RATES[i]), float(sc[i]))); a = np.searchsorted(RATES, RATES[i] * 0.97); b = np.searchsorted(RATES, RATES[i] * 1.03); s2[a:b] = 0
    return out

def chains(P, gap=3, shortest=6):
    """Peaks that follow one another as a moving copy's would: the next peak sits where this one's stretch says the delay
    will be. Returns a list of index arrays into P (each a chain of at least `shortest` peaks, frames rising)."""
    order = np.argsort(P[:, 0], kind='stable'); open_ = []; done = []                 # open chains: [indices]
    by_frame = {}
    for i in order: by_frame.setdefault(int(P[i, 0]), []).append(i)
    for t in sorted(by_frame):
        cand = by_frame[t]; used = set(); pairs = []
        for ci, ch in enumerate(open_):
            l = ch[-1]; dt = t - P[l, 0]
            for i in cand:
                rho = 0.5 * (P[l, 2] + P[i, 2]); err = abs(P[i, 1] - (P[l, 1] + rho * dt * HOP))
                if err < TOL * (1 + 0.5 * (dt - 1)) and abs(P[i, 2] - P[l, 2]) < 0.006 * dt + 0.002: pairs.append((err, ci, i))
        taken = set()
        for err, ci, i in sorted(pairs):
            if ci in taken or i in used: continue
            open_[ci].append(i); taken.add(ci); used.add(i)
        keep = []
        for ci, ch in enumerate(open_):
            if t - P[ch[-1], 0] >= gap and ci not in taken: done.append(ch)
            else: keep.append(ch)
        open_ = keep + [[i] for i in cand if i not in used]
    done += open_
    return [np.array(c) for c in done if len(c) >= shortest]

SLOPE = 0.5 * TOL / RTOL                                     # how a miss in stretch counts against a miss in delay
def _solve(P, idx, rate, kind, ph=0.0):
    """The curve of this rate (and, for a triangle, phase) through the peaks idx: delay and stretch fitted together."""
    t = P[idx, 0] / FPS; w = np.sqrt(P[idx, 3]); om = 2 * np.pi * rate
    if kind == 0:
        s, c = np.sin(om * t), np.cos(om * t); B = np.concatenate([np.stack([np.ones(len(t)), s, c], 1) * w[:, None], np.stack([np.zeros(len(t)), om * c / FS, -om * s / FS], 1) * (SLOPE * w)[:, None]])
    else:
        s, ds = shape(rate * t + ph, 1); B = np.concatenate([np.stack([np.ones(len(t)), s], 1) * w[:, None], np.stack([np.zeros(len(t)), ds * rate / FS], 1) * (SLOPE * w)[:, None]])
    y = np.concatenate([P[idx, 1] * w, P[idx, 2] * SLOPE * w]); sol = np.linalg.lstsq(B, y, rcond=None)[0]
    if kind == 0: return dict(kind=0, rate=float(rate), phase=float(np.arctan2(sol[2], sol[1]) / (2 * np.pi)), mid=float(sol[0]), half=float(np.hypot(sol[1], sol[2])))
    return dict(kind=1, rate=float(rate), phase=float(ph), mid=float(sol[0]), half=float(sol[1]))

def _small(A, b):
    """A few equations by elimination (as the app's code does it); None if they cannot be solved."""
    A = np.array(A, float); b = np.array(b, float); m = len(b); big = np.abs(A).max()
    if not (big > 0.0) or not np.isfinite(big): return None
    for c in range(m):
        p = c + int(np.argmax(np.abs(A[c:, c])))
        if abs(A[p, c]) < 1e-13 * big: return None
        if p != c: A[[p, c]] = A[[c, p]]; b[[p, c]] = b[[c, p]]
        for r in range(c + 1, m): f = A[r, c] / A[c, c]; A[r, c:] -= f * A[c, c:]; b[r] -= f * b[c]
    for c in range(m - 1, -1, -1): b[c] = (b[c] - A[c, c + 1:] @ b[c + 1:]) / A[c, c]
    return b if np.all(np.isfinite(b)) else None

def _refine(P, idx, v):
    """All four numbers of the curve (also its rate) fitted to the peaks idx, far-off peaks counting less (a miss of e counts
    as 2 (sqrt(1 + (e / 0.45)^2) - 1)). Step by step as the app's code does it: every step solves the weighted equations of
    the curve's slopes, held back as far as needed for the misses to shrink."""
    t = P[idx, 0] / FPS; w = np.sqrt(P[idx, 3]); pd = P[idx, 1]; pr = P[idx, 2]; scale = np.array([1.0, 1.0, 0.01, 0.002]); fs = TOL * 0.3; kind = v['kind']; n = len(t)
    def ev(x, jac):
        s, ds = shape(x[3] * t + x[2], kind); f = np.empty(2 * n); f[0::2] = (x[0] + x[1] * s - pd) * w; f[1::2] = (x[1] * ds * x[3] / FS - pr) * SLOPE * w
        cost = 0.5 * fs * fs * float((2.0 * (np.sqrt(1.0 + (f / fs) ** 2) - 1.0)).sum())
        if not jac: return cost, f, None
        dds = -4 * np.pi ** 2 * s if kind == 0 else np.zeros(n); J = np.zeros((2 * n, 4)); J[0::2, 0] = w; J[0::2, 1] = s * w; J[0::2, 2] = x[1] * ds * w; J[0::2, 3] = x[1] * ds * t * w
        J[1::2, 1] = ds * x[3] / FS * SLOPE * w; J[1::2, 2] = x[1] * dds * x[3] / FS * SLOPE * w; J[1::2, 3] = x[1] * (dds * t * x[3] + ds) / FS * SLOPE * w
        return cost, f, J * scale
    x = np.array([v['mid'], v['half'], v['phase'], v['rate']], float); cost, f, J = ev(x, True); lam = 1e-3
    if not np.isfinite(cost): return v
    for _ in range(60):
        wi = 1.0 / np.sqrt(1.0 + (f / fs) ** 2); H = (J * wi[:, None]).T @ J; g = J.T @ (wi * f); better = False
        for _ in range(12):
            d = _small(H + np.diag(lam * np.diag(H) + 1e-300), -g)
            if d is not None:
                x2 = x + scale * d; c2 = ev(x2, False)[0]
                if np.isfinite(c2) and c2 < cost: better = True; break
            lam *= 10.0
        if not better: break
        gain = cost - c2; step = float(np.abs(d).max()); x = x2; cost, f, J = ev(x, True); lam = max(lam * 0.1, 1e-12)
        if gain < 1e-13 * cost or step < 1e-9: break
    if not (0.05 < x[3] < 5.0): return v
    out = dict(v); out.update(mid=float(x[0]), half=float(x[1]), phase=float(x[2] % 1.0), rate=float(x[3]))
    if out['half'] < 0 and out['kind'] == 0: out['half'] = -out['half']; out['phase'] = (out['phase'] + 0.5) % 1.0
    return out

LEAST = 0.1                                                  # a real voice shows on the whole of its round: in every eighth of the round this share of the frames with peaks have one on its curve
TWIN = 0.4                                                   # the sound's own pitch has twins (peaks at simple ratios of its delay) in this share of its peaks; a chorus voice stands alone
STEEP = 0.056                                                # a curve steeper than this cannot be seen (the stretches looked at end at 6 %)
RATIOS = ((2, 1), (3, 1), (4, 1), (1, 2), (1, 3), (1, 4), (3, 2), (2, 3), (4, 3), (3, 4), (5, 2), (2, 5), (5, 3), (3, 5), (5, 4), (4, 5))

def cover(P, v):
    """Per eighth of the voice's round: the share of the frames with peaks that have one on the curve (-1: cannot be said,
    under 3 frames there or the curve too steep there to be seen), and the least of them (0 if fewer than six can be said).
    The sound's own pitch drifting, or a drum's falling tone, lies along one flank of a curve at best; a real voice is there all the way round."""
    fr = np.unique(P[:, 0]); hit = np.unique(P[near(P, v), 0]); k_all = np.minimum((((v['rate'] * fr / FPS + v['phase']) % 1.0) * 8.0).astype(int), 7); k_hit = np.minimum((((v['rate'] * hit / FPS + v['phase']) % 1.0) * 8.0).astype(int), 7)
    out = []
    for k in range(8):
        a = int((k_all == k).sum()); slope = abs(v['half'] * float(shape((k + 0.5) / 8.0, v['kind'])[1]) * v['rate'] / FS)
        out.append(-1.0 if a < 3 or slope > STEEP else float((k_hit == k).sum()) / a)
    said = [c for c in out if c >= 0.0]; return out, (min(said) if len(said) >= 6 else 0.0)

def twin(P, v):
    """The share of the voice's peaks with a twin: another peak in the same frame at a simple ratio of its delay, the stretch
    in the same ratio (the largest share over the ratios; a ratio counts if it can be looked at for 30 % of the peaks). At
    whole multiples only a twin at least three quarters as high counts: a voice that is fed back has a fainter one at twice its delay."""
    m = near(P, v); idx = np.where(m)[0]; top = 0.0
    if len(idx) == 0: return 0.0
    first = np.searchsorted(P[:, 0], P[idx, 0], 'left'); last = np.searchsorted(P[:, 0], P[idx, 0], 'right')
    for a, b in RATIOS:
        f = a / b; k = 2.0 * max(f, 1.0); can = 0; got = 0
        for i, lo, hi in zip(idx, first, last):
            d = P[i, 1] * f; r = P[i, 2] * f
            if d < TMIN + 2 or d > TMAX - 2 or abs(r) > STEEP: continue
            can += 1
            for j in range(lo, hi):
                if j != i and abs(P[j, 1] - d) < k * TOL and abs(P[j, 2] - r) < k * RTOL and (b != 1 or P[j, 3] >= 0.75 * P[i, 3]): got += 1; break
        if can > 0 and can >= 0.3 * len(idx): top = max(top, got / can)
    return top

def fits(v, T):
    """Moves (0.08 ms each way at least), stays inside the delays looked at, goes round one and a half times within the sound."""
    return v is not None and abs(v['half']) >= 2.0 and v['mid'] - abs(v['half']) >= TMIN and v['mid'] + abs(v['half']) <= TMAX and v['rate'] * T / FPS >= 1.5

def _group(P, T, use, seeds, rate, kind, most, frames_with, need):
    """The voices of one kind at one rate: every chain of peaks gives a curve; the curve that collects the most (peak heights
    near it) is a voice, its peaks are taken away, and again. Returns (collected in all, [(voice, peak mask)])."""
    cands = []
    for seed in seeds:
        if kind == 0: v = _solve(P, seed, rate, 0)
        else:
            best = None                                                   # a chain lies on one slope of a triangle and cannot say where the triangle turns: every place is tried (coarsely with a wide catch, then finely), the one that collects the most peaks is taken
            for stage, phases, k in ((0, np.arange(64) / 64.0, 3.0), (1, None, 1.0)):
                if stage == 1:
                    if best is None: break
                    phases = best[1]['phase'] + np.arange(-8, 9) / 512.0; best = None
                for ph in phases:
                    q = _solve(P, seed, rate, 1, ph % 1.0)
                    if q['half'] < 0: q['half'] = -q['half']; q['phase'] = (q['phase'] + 0.5) % 1.0
                    if not fits(q, T): continue
                    e = P[near(P, q, k) & use, 3].sum()
                    if best is None or e > best[0]: best = (e, q)
            v = best[1] if best else None
        if fits(v, T): cands.append((v, near(P, v) & use))
    out = []; total = 0.0; free = np.ones(len(P), bool)
    for _ in range(most):
        if not cands: break
        got = [P[m & free, 3].sum() for v, m in cands]; i = int(np.argmax(got)); v, m = cands[i]; m = m & free
        for _ in range(2):                                                 # fitted to what it collects, and collected again
            if m.sum() < 8: break
            q = _refine(P, np.where(m)[0], v)
            if not fits(q, T): break
            v = q; m = near(P, v) & use & free
        if len(np.unique(P[m, 0])) < need * frames_with: break
        out.append((v, m)); total += P[m, 3].sum(); free &= ~m
    return total, out

def voices(P, T, most=4, need=0.25, info=None):
    """Voices through the peaks P of T frames: dicts(kind, rate, phase, mid, half, on = share of the sounding frames with a
    peak on the curve, height = their mean height). For each rate at which the peaks come round (rounds) and each kind (sine,
    triangle) the voices of that rate and kind are drawn out together; the rate and kind that explain the most are kept,
    their peaks taken away, and the search goes on for voices of another rate."""
    C = [c for c in chains(P, shortest=5) if np.ptp(P[c, 1]) >= 2.0]; frames_with = len(np.unique(P[:, 0])); found = []; use = np.ones(len(P), bool)       # chains that move: one that stands still is the sound's own pitch
    if not C or len(P) < 20: return found
    for _ in range(3):
        if len(found) >= most: break
        seeds = sorted([c[use[c]] for c in C if use[c].sum() >= 5], key=lambda c: -P[c, 3].sum())[:60]
        if not seeds: break
        idx = np.unique(np.concatenate(seeds))
        best = None
        for rate, rs in rounds(P[idx], T, most=10):                         # the rates come from the moving chains' peaks alone
            for kind in (0, 1):
                total, grp = _group(P, T, use, seeds, rate, kind, most - len(found), frames_with, need); total *= 1.0 if kind == 0 else 0.95       # a triangle has to be clearly the better one
                if grp and (best is None or total > best[0]): best = (total, grp, rs)
        if best is None: break
        for v, m in best[1]:
            for _ in range(3):
                mm = near(P, v) & use
                if mm.sum() < 8: break
                v = _refine(P, np.where(mm)[0], v)
            if not fits(v, T): continue
            mm = near(P, v) & use; v.update(on=len(np.unique(P[mm, 0])) / max(frames_with, 1), height=float(P[mm, 3].mean()) if mm.any() else 0.0, piled=best[2])
            v['cover'], v['least'] = cover(P, v); v['twin'] = twin(P, v)
            if info is not None: info.append(dict(v))
            if v['on'] >= need and v['least'] >= LEAST and v['twin'] < TWIN: found.append(v)
            use &= ~near(P, v, 1.5)
    return found
