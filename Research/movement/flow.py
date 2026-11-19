# The reader of pitch movement (what the PITCH fader of the MOVEMENT group works with): track(x) gives how a sound's
# pitch moves, in cents every 10 ms; where nothing can be followed the curve stands still.
# Pitch movement is read off a sound without naming its pitch: how far the lines of the spectrum (the partials) shift
# from one moment to another, on a logarithmic frequency axis, measured between moments 10 ms to 1.3 s apart and put
# together into one curve by least squares. The notes of a tune are not followed (a step between two notes is no shift
# of lines that can be matched); bends, glides, vibrato, revs and Doppler are.
# Two looks: a short one (85 ms windows) that follows fast movement, and a long one (341 ms) that tells apart partials
# 12 Hz from each other, which an engine at low revs needs.
# Rounds: the sound is flattened with the whole curve read so far (tape speed, warp.py) and read again (its lines then
# stand still, so the long window can tell close partials apart even where the pitch moves fast), until nothing more is
# left to read; then once more, strictly. Where a round follows nothing, the curve the sound is flattened with keeps the
# movement it had. What the strict reading still finds on top says where the curve is not sure: there it is not kept.
# "Followed": the same lines are found again over a span of time, directly, between moments whose windows do not overlap
# (two windows that overlap are alike even in noise). A sound that is lost for a moment (up to 0.6 s) is looked for on
# the other side of the loss, from where it lies if it stood still to where it lies if it went on moving. Found, the
# curve goes through the loss in a straight line, if that is proven: the lines have to stand stiller with the glide than
# with the same step taken at once (a sound's lines fit those of its own harmonics in part, and are found where they
# are not). Not found but most alike near there, the sound goes on and the curve stands still across the loss; most
# alike far away, or found and not proven, another sound may start. A pause (the level 30 dB under what sounds before
# and after) separates two sounds for good.
# What the reader cannot read it leaves alone. The curve that is kept stands still wherever the strict reading follows
# nothing, and the next stretch goes on where the last one left it; it begins afresh only at a pause. A step anywhere
# else has to be proven: where no sound is sustained (zaps, chirps) a short event keeps its place only if the lines
# after the step are alike those before it once the sound is flattened with the curve, and beside a sustained sound a
# short event is not kept at all (see compose, across). No reading follows a step that an earlier round put into the
# curve the sound is flattened with. The curve's fast part (above about 3 Hz) is kept only where the lines stand
# stiller with it than without (see proven): two sounds close in pitch beat, and a curve that follows the flutter of
# their lines would add one. Movement that is not read stays in the sound.
# What it cannot read: noise (no lines), partials closer than about 25 Hz that move fast, falling or rising sounds that
# follow each other without a pause (read in part), short sounds beside a sustained one, two things that move
# differently at the same time (it follows one of them, or neither).
# Where a sound is cut off hard (the ends of nearly every file): a low sound's curve stands still over the 40 ms next to
# the cut, where the reading would run away (see cuts); a sound whose lines lie high is read right up to the cut.
# The app's port: Source/Core/Flow.h (Tools/move_parity.py compares the two). Test sounds and scores: test_flow.py.
import os, sys, numpy as np, scipy.fft as sfft, scipy.ndimage as ndi, scipy.linalg as sl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp

SR = 48000
STEP = 480                          # the curve: a value every 10 ms
LOOKS = ((4096, 480), (16384, 960))  # window, hop
FMIN = 35.0; FMAX = 8000.0; RES = 5.0       # the log axis: 5 cents per step
NB = int(np.ceil(1200 * np.log2(FMAX / FMIN) / RES)) + 1
SPAN = 1.3                          # s: the longest span measured directly
RANGE = 70.0                        # cents searched either way between neighbouring frames
NEAR = 20.0                         # cents searched around what the curve so far says, for frames further apart
SIGMA = 1.2                         # steps: the width a line is drawn with
OVER = 10.0                         # dB over the noise around it that makes a peak a line
FLOOR_HZ = 375.0                    # the noise around a bin: from the lower quarter of the bins in a block this wide
AGREE = 4.0                         # cents: a measurement agrees with the curve when it is this close (plus a tenth of its size)
PK = 0.5                            # how alike two frames' lines must be for a measurement to count at all
TRUST = 0.75                        # share of a moment's measurements that must agree for the moment to be held
EDGE = 0.05                         # the long look takes over when its measurements agree by this much more than the short look's

def frames(x, n, hop):
    xp = np.concatenate([np.zeros(n // 2), x, np.zeros(n)]); T = 1 + len(x) // hop
    return xp[np.arange(T)[:, None] * hop + np.arange(n)[None, :]]

def lines(x, n, hop):
    """(T, NB): every frame's lines on the log axis. A line is a peak of the spectrum that stands OVER dB over the noise
    around it; it is drawn at the frequency its phase says it has (not at its bin), the more over the noise the stronger."""
    F = frames(x, n, hop); i = np.arange(n); h = 0.5 - 0.5 * np.cos(2 * np.pi * i / n); dh = (np.pi / n) * np.sin(2 * np.pi * i / n)
    T = len(F); K = n // 2 + 1; A = np.zeros((T, NB + 8)); e = (F ** 2).mean(1); cnt = np.zeros(T, int)
    blk = max(8, int(round(FLOOR_HZ * n / SR))); nblk = K // blk; ctr = (np.arange(nblk) + 0.5) * blk; kk = np.arange(K)
    j = np.clip(np.searchsorted(ctr, kk) - 1, 0, nblk - 2); u = np.clip((kk - ctr[j]) / blk, 0.0, 1.0); w = 2 * np.pi * kk / n
    for a in range(0, T, 256):          # in batches (memory)
        X = sfft.rfft(F[a:a + 256] * h, axis=1); D = sfft.rfft(F[a:a + 256] * dh, axis=1); P = np.abs(X) ** 2; t_ = len(P)
        Q = np.percentile(P[:, :nblk * blk].reshape(t_, nblk, blk), 25, axis=2) / 0.288        # noise: the lower quartile is 0.288 of its mean
        floor = np.maximum(Q[:, j] * (1 - u) + Q[:, j + 1] * u, 1e-10 * P.max() + 1e-30)
        inst = w[None, :] - np.imag(D * np.conj(X)) / np.maximum(P, 1e-30); f = inst * SR / (2 * np.pi)
        pk = np.zeros_like(P, bool); pk[:, 1:-1] = (P[:, 1:-1] > P[:, :-2]) & (P[:, 1:-1] >= P[:, 2:])
        over = 10 * np.log10(np.maximum(P, 1e-30) / floor)
        ok = pk & (over > OVER) & (np.abs(inst - w[None, :]) < 1.5 * 2 * np.pi / n) & (f > FMIN) & (f < FMAX)
        t = a + np.repeat(np.arange(t_), ok.sum(1)); pos = 1200 * np.log2(f[ok] / FMIN) / RES; cnt[a:a + t_] = ok.sum(1)
        wt = np.clip((over[ok] - OVER + 4.0) / 24.0, 0.0, 1.0); i0 = np.round(pos).astype(np.int64)
        for o in range(-4, 5):
            b = i0 + o; g = (b >= 0) & (b < NB)
            np.add.at(A, (t[g], b[g]), wt[g] * np.exp(-0.5 * ((b[g] - pos[g]) / SIGMA) ** 2))
    return A[:, :NB], e, cnt

def match(A, nrm, d, centre, S):
    """Frames t and t+d: the shift (cents, searched around centre[t] steps) at which their lines agree best, and how well (0..1)."""
    T = len(A); n = T - d; a = A[:n]; o = np.round(centre).astype(np.int64); C = np.zeros((n, 2 * S + 1)); idx = np.arange(NB)[None, :]
    for s in range(-S, S + 1):
        j = idx + (o + s)[:, None]; g = (j >= 0) & (j < NB)
        C[:, s + S] = (a * np.where(g, np.take_along_axis(A[d:], np.clip(j, 0, NB - 1), 1), 0.0)).sum(1)
    C /= (nrm[:n] * nrm[d:] + 1e-12)[:, None]
    j = C.argmax(1); r = np.arange(n); pk = C[r, j]; inside = (j > 0) & (j < 2 * S)
    jj = np.clip(j, 1, 2 * S - 1); l0, l1, l2 = (np.log(np.maximum(C[r, jj + q], 1e-9)) for q in (-1, 0, 1))
    den = l0 - 2 * l1 + l2; off = np.where(den < -1e-9, 0.5 * (l0 - l2) / np.minimum(den, -1e-9), 0.0)
    return (o + jj - S + np.clip(off, -1, 1)) * RES, pk, inside

def solve(T, I, J, S, W0, lam0=1e-6, lam1=1e-4, huber=6.0, rounds=3):
    """The curve p (cents) that fits the links best: sum w (p[j]-p[i]-s)^2 + lam1 sum (p[t+1]-p[t])^2 + lam0 sum p^2."""
    p = np.zeros(T)
    if len(I) == 0: return p
    bw = int((J - I).max()); W = W0.copy()
    for it in range(rounds):
        ab = np.zeros((bw + 1, T)); rhs = np.zeros(T)       # upper banded: ab[bw + i - j, j] = M[i, j]
        np.add.at(ab[bw], I, W); np.add.at(ab[bw], J, W); np.add.at(ab, (bw - (J - I), J), -W)
        np.add.at(rhs, J, W * S); np.add.at(rhs, I, -W * S)
        ab[bw] += lam0; ab[bw, :-1] += lam1; ab[bw, 1:] += lam1; ab[bw - 1, 1:] -= lam1
        p = sl.solveh_banded(ab, rhs)
        r = np.abs(p[J] - p[I] - S); W = W0 * np.minimum(1.0, huber / np.maximum(r, 1e-9))
    return p

TIE = 0.35          # how alike two moments' lines must be to count as the same lines (at the place the curve says)
COVER = 0.5         # share of the pairs across a moment that must be tied for the moment to count as followed over that span
LONG = 0.585        # s: lines followed over this span or more: the chain is trusted as far as it goes
CUT = 0.25          # inside a stretch, a place where pairs reach less than this share of what they reach on both sides of it: the lines before and after are not the same
REACH = 2.0         # where lines are followed over short spans only, the chain counts against its average over this many times the span
DONE = (1.0, 8.0)   # cents (rms, range) left to read at which the flattening has settled
PASSES = 5          # at most
GAP = 0.25          # s: two stretches up to this far apart are joined by a straight line
RAMP = 0.1          # s: over which the curve goes from an event to the level held around it, and comes to the next
JOIN = 0.6          # s: a sound lost for at most this long is looked for again on the other side of the loss
WIDE = 240.0        # cents: how far from what the movement on both sides suggests it is looked for at most
FAR = 1300.0        # cents: how far it is looked to see whether what is most alike lies near or far
WEAK = 0.25         # how alike the two sides of a loss must be at least for the sound to count as going on
CRUMB = 0.15        # s: a stretch shorter than this next to a short loss hangs onto its neighbour
SURE = (3.0, 0.5)   # cents, share: where reading the flattened sound again still finds more than this (and more than this share of the curve's own movement), the curve is not kept
PAUSE = 1e-3        # a pause: the level this far under what sounds before and after it
JUMP = 1e-4         # a cut: the level rises or falls by this much (40 dB) within two hundredths of a second; the ends of the file are cuts
NEARCUT = 1632      # samples (3.4 frames): this near a cut the short window has a tenth of its weight or more at the cut
REACHCUT = 8192     # samples: from this far outside a sound the long window still reaches it
SHIFT = (2.0, 0.5)  # Hz: how far a cut through its window moves a line on average, for the short look and the long one
STEADY = 1.1        # a sound found again across a loss: the curve goes through the loss only if the lines then stand this many times stiller than with a curve that takes the same step at once
AROUND = 0.6        # s: how much of the sound on both sides of the loss is flattened for that
SLOW = 6.0          # moments: the curve smoothed this much is its slow part (below about 3 Hz); the rest is its fast part
FASTKEEP = (1.0, 1.05)  # the fast part is dropped where the lines stand no stiller with it than with the slow part alone, and kept whole where they stand this many times stiller
FINE = 2            # the spectrum the stillness of lines is measured in: this many values per bin of the long window (the measure then does not depend on where between two bins a line lies)
LINES = 3.0         # a block of the spectrum holds lines when its energy is gathered this many times closer than evenly spread energy is (noise: 2)
DUST = 1e-6         # of a frame's mean energy per value: added to every value, so that what is left of nothing does not count as a line
GHOP = 3840         # samples: how far apart the frames lie in which the stillness of lines is measured (80 ms; the long window is 341 ms long)
STAND = 7.0         # cents: beside a cut a look's reading counts only where that moves the sound's lines by less than this on average
ALIKE = 0.75        # a step of the curve between two stretches is proven when, the sound flattened with the curve, the lines after the step are this alike those before it
ROUND = 150.0       # cents: how far from where the step puts them the lines after it are looked for (they are then put where they are found)

def agrees(p, I, J, S, k=1.0): return np.abs(p[J] - p[I] - S) < k * (AGREE + 0.1 * np.abs(S))

def trust_of(T, p, I, J, S, W, hop, k=1.0):
    good = np.zeros(T); allw = np.zeros(T); ok = agrees(p, I, J, S, k)
    for K in (I, J): np.add.at(good, K, W * ok); np.add.at(allw, K, W)
    q = max(1, int(round(0.1 * SR / hop))) | 1; g = ndi.uniform_filter1d(good, q, mode='nearest'); a = ndi.uniform_filter1d(allw, q, mode='nearest')
    return g / (a + 1e-9) * np.minimum(1.0, g / 0.5)

def match_at(A, nrm, i, d, centre, S):
    """As match, for the frames i only; and the whole table of likenesses."""
    a = A[i]; b = A[i + d]; o = np.round(centre).astype(np.int64); C = np.zeros((len(i), 2 * S + 1)); idx = np.arange(NB)[None, :]
    for s in range(-S, S + 1):
        j = idx + (o + s)[:, None]; g = (j >= 0) & (j < NB)
        C[:, s + S] = (a * np.where(g, np.take_along_axis(b, np.clip(j, 0, NB - 1), 1), 0.0)).sum(1)
    C /= (nrm[i] * nrm[i + d] + 1e-12)[:, None]
    j = C.argmax(1); r = np.arange(len(i)); pk = C[r, j]; inside = (j > 0) & (j < 2 * S)
    jj = np.clip(j, 1, 2 * S - 1); l0, l1, l2 = (np.log(np.maximum(C[r, jj + q], 1e-9)) for q in (-1, 0, 1))
    den = l0 - 2 * l1 + l2; off = np.where(den < -1e-9, 0.5 * (l0 - l2) / np.minimum(den, -1e-9), 0.0)
    return (o + jj - S + np.clip(off, -1, 1)) * RES, pk, inside, C

def look(x, n, hop, loose=1.0, trust=TRUST, ties=True, speed=None, log=None, brk=None, cv=None):
    A, e, cnt = lines(x, n, hop); T = len(A); nrm = np.sqrt((A ** 2).sum(1)); live = (cnt >= 1) & (e > e.max() * 1e-6)
    cb = np.concatenate([[0], np.cumsum(brk if brk is not None else np.zeros(T, bool))])
    def free(d): return (cb[d + 1:T + 1] - cb[:T - d]) == 0          # pairs (t, t + d) with no break between two sounds inside
    I = np.zeros(0, np.int64); J = I.copy(); S = np.zeros(0); W = np.zeros(0); p = np.zeros(T); d = 1
    It = I.copy(); Jt = I.copy(); St = S.copy()
    while d < T and d * hop <= SPAN * SR:
        first = d * hop <= 0.021 * SR; centre = np.zeros(T - d) if first else (p[d:] - p[:-d]) / RES
        sh, pk, inside = match(A, nrm, d, centre, int(round((RANGE if first else NEAR) / RES)))
        fr = free(d); g = inside & live[:-d] & live[d:] & (pk > PK) & fr; i = np.nonzero(g)[0]
        I = np.concatenate([I, i]); J = np.concatenate([J, i + d]); S = np.concatenate([S, sh[i]]); W = np.concatenate([W, ((pk[i] - PK) / (1 - PK)) ** 2 / np.sqrt(d)])
        if d * hop >= 0.9 * n:
            i = np.nonzero(inside & live[:-d] & live[d:] & (pk > TIE) & fr)[0]; It = np.concatenate([It, i]); Jt = np.concatenate([Jt, i + d]); St = np.concatenate([St, sh[i]])
        p = solve(T, I, J, S, W); d *= 2
    tr = trust_of(T, p, I, J, S, W, hop, loose)
    for it in range(2):
        keep = (tr[I] >= trust) & (tr[J] >= trust)
        if keep.all(): break
        I, J, S, W = I[keep], J[keep], S[keep], W[keep]; p = solve(T, I, J, S, W); tr = np.minimum(tr, trust_of(T, p, I, J, S, W, hop, loose))
    span = np.zeros(T); reach = np.zeros(T); tls = [d for d in (2 ** np.arange(12)) if d * hop >= 0.9 * n and d < T and d * hop <= SPAN * SR]
    if ties and len(It) and len(tls):
        def covering(i, dd):            # per frame: how many of the pairs (i, i + dd) lie across it
            c = np.zeros(T + 1); np.add.at(c, i, 1); np.add.at(c, i + dd + 1, -1); return np.cumsum(c)[:T]
        d0 = tls[0]; a0 = np.arange(T - d0); cand0 = a0[live[:-d0] & live[d0:]]; nc0 = covering(cand0, d0); mended = np.zeros(T, bool); mlag = np.zeros(T); goes = np.zeros(T, bool); other = np.zeros(T, bool)
        def spans(It, Jt, St):
            """Which moments are followed over the shortest span (base), the longest span over each moment, how far to both sides."""
            ok = agrees(p, It, Jt, St, 2.0); It, Jt = It[ok], Jt[ok]; span = np.zeros(T); reach = np.zeros(T)
            base = (nc0 > 0) & (covering(It[(Jt - It) == d0], d0) >= COVER * nc0)
            for dd in tls:
                i = It[(Jt - It) == dd]; a = np.arange(T - dd)
                if dd == d0: got = base
                else:
                    cand = a[base[:-dd] & base[dd:]]; i = i[base[i] & base[i + dd]]     # longer spans: of the pairs of moments followed over the shortest
                    nc = covering(cand, dd); nt = covering(i, dd); got = (nc > 0) & (nt >= COVER * nc) & (base | mended)
                span[got] = dd * hop / SR
                if len(i):
                    tie = np.zeros(T, bool); tie[i] = True; dist = ndi.distance_transform_edt(~tie); h = dd // 2
                    r = np.zeros(T); r[h:] = np.maximum(0.0, dd / 2.0 - dist[:T - h]); reach = np.maximum(reach, r * hop / SR)
            span = np.maximum(span, mlag); reach = np.maximum(reach, 0.5 * mlag)         # a loss the sound was found again across: followed, over the span it was found across
            return base, span, reach
        def again():                    # the pairs once more, around the mended curve
            It = np.zeros(0, np.int64); Jt = It.copy(); St = np.zeros(0)
            for d in tls:
                sh, pk, inside = match(A, nrm, d, (p[d:] - p[:-d]) / RES, int(round(NEAR / RES)))
                i = np.nonzero(inside & live[:-d] & live[d:] & (pk > TIE) & free(d))[0]; It = np.concatenate([It, i]); Jt = np.concatenate([Jt, i + d]); St = np.concatenate([St, sh[i]])
            return It, Jt, St
        def mend(weak, base, what):
            """A sound lost or followed badly for a moment (the frames weak): is it the same sound on the other side? Measured
            directly, from moments before to moments after, as far from what the movement on both sides suggests as that movement is fast."""
            idx = np.nonzero(weak)[0]; any_ = False
            for r in (np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []):
                a, b = r[0], r[-1] + 1; L = b - a
                if a < 3 or b > T - 3 or L * hop > JOIN * SR or not base[a - 1] or not base[b]: continue
                ls = a - 1
                while ls > 0 and base[ls - 1] and not weak[ls - 1]: ls -= 1
                re_ = b + 1
                while re_ < T and base[re_] and not weak[re_]: re_ += 1
                K = max(2, int(round(0.2 * SR / hop))); la = max(ls, a - 1 - K); rb = min(re_, b + 1 + K)
                if a - la < 3 or rb - b < 3 or cb[rb] - cb[la] > 0: continue
                if e[a:b].min() < 1e-3 * min(e[la:a].mean(), e[b:rb].mean()): continue          # a pause is no loss
                sl = (p[a - 1] - p[la]) / (a - 1 - la); sr_ = (p[rb - 1] - p[b]) / (rb - 1 - b)
                V = max(np.sqrt((np.diff(p[la:a]) ** 2).mean()), np.sqrt((np.diff(p[b:rb]) ** 2).mean()))
                if speed is not None: V = max(V, float(speed[max(0, la):rb].max()) * hop / SR)
                D0 = (L + 1) * 0.5 * (sl + sr_); H = min(WIDE, 2 * NEAR + (L + 1) * V); Dp = D0
                if cv is not None: D0 += (L + 1) * 0.5 * ((cv[a - 1] - cv[la]) / (a - 1 - la) + (cv[rb - 1] - cv[b]) / (rb - 1 - b)) - (cv[b] - cv[a - 1])     # the sound was flattened: where the whole curve's movement on both sides points, less what the flattening took
                Ds = -(cv[b] - cv[a - 1]) if cv is not None else 0.0                    # and where it lies if the sound stood still across the loss
                lo_, hi_ = min(D0, Ds) - H, max(D0, Ds) + H; D0 = 0.5 * (lo_ + hi_); H = min(0.5 * (hi_ - lo_), 0.75 * FAR)       # looked for from the one to the other: a movement that stops inside the loss ends in between
                done = False; seen = None
                for d in tls:
                    i = np.arange(max(ls, b + 1 - d), min(a - 2, re_ - 1 - d) + 1)
                    if len(i) < 3: continue
                    pre = p[a - 1] - p[i]; post = p[i + d] - p[b]; Sf = int(round(FAR / RES)); Sh = int(round(H / RES))
                    sh, pk, inside, C = match_at(A, nrm, i, d, (pre + D0 + post) / RES, Sf)
                    Cn = C[:, Sf - Sh:Sf + Sh + 1]; j = Cn.argmax(1); r_ = np.arange(len(i)); pkn = Cn[r_, j]; ins = (j > 0) & (j < 2 * Sh)        # the best place near where the movement points
                    jj = np.clip(j, 1, 2 * Sh - 1); l0, l1, l2 = (np.log(np.maximum(Cn[r_, jj + q_], 1e-9)) for q_ in (-1, 0, 1))
                    den = l0 - 2 * l1 + l2; off = np.where(den < -1e-9, 0.5 * (l0 - l2) / np.minimum(den, -1e-9), 0.0); shn = (np.round((pre + D0 + post) / RES) + jj - Sh + np.clip(off, -1, 1)) * RES
                    Cm = C.mean(0); near = np.abs(np.arange(-Sf, Sf + 1)) <= Sh; here = (float(Cm[near].max()), float(Cm[~near].max()))
                    if seen is None: seen = here
                    g = ins & (pkn > PK)
                    if g.sum() < max(3, 0.5 * len(i)) or here[0] < here[1]: continue          # (what is more alike elsewhere is not this sound found again: lines of a sound fit those of its own harmonics in part)
                    br = (shn - pre - post)[g]; M = np.median(br); okk = np.abs(br - M) < 2 * AGREE + 0.1 * abs(M)
                    if okk.sum() < max(3, 0.6 * g.sum()): continue
                    M = br[okk].mean(); old = p[b] - p[a - 1]; done = True
                    if what == 'weak' and abs(M - old) < 2 * AGREE: break            # as the curve has it already
                    p[a:b] = p[a - 1] + M * (np.arange(L) + 1.0) / (L + 1.0); p[b:] += M - old; mended[a:b] = True; mlag[a:b] = d * hop / SR; any_ = True
                    if log: log('   %s %.2f to %.2f s, over %.2f s: %.0f cents instead of %.0f (searched %.0f +- %.0f, of which %.0f from what is left, alike %.2f)' % (what, a * hop / SR, b * hop / SR, d * hop / SR, M, old, D0, H, Dp, pkn[g].mean()))
                    break
                # not found again for sure: does the sound go on all the same (what is most alike lies near where the movement points), or does another start?
                if not done and what == 'lost' and seen is not None and seen[0] >= WEAK and seen[0] >= seen[1] - 0.1:
                    goes[a:b] = True
                    if log: log('   lost %.2f to %.2f s: goes on (alike %.2f near, %.2f far)' % (a * hop / SR, b * hop / SR, seen[0], seen[1]))
                elif not done and what == 'lost' and seen is not None:
                    other[a:b] = True
                    if log: log('   lost %.2f to %.2f s: another sound (alike %.2f near, %.2f far)' % (a * hop / SR, b * hop / SR, seen[0], seen[1]))
            return any_
        base, span, reach = spans(It, Jt, St)
        if mend(~base, base, 'lost'):
            It, Jt, St = again(); base, span, reach = spans(It, Jt, St)
        # a place where pairs reach much less far than on both sides of it
        m = np.minimum(np.maximum.accumulate(reach), np.maximum.accumulate(reach[::-1])[::-1]); low = base & (m > 0) & (reach <= CUT * m)
        if low.any() and mend(low, base, 'weak'):
            It, Jt, St = again(); base, span, reach = spans(It, Jt, St)
        tr = np.where(mended, np.maximum(tr, trust), tr)
    ok_ = ties and len(It) and len(tls)
    return p, tr, span, reach, e, cnt, (mended if ok_ else np.zeros(T, bool)), (goes if ok_ else np.zeros(T, bool)), (other if ok_ else np.zeros(T, bool))

def both(x, loose=1.0, trust=TRUST, ties=True, speed=None, log=None, brk=None, cv=None):
    """Both looks on the 10 ms grid of x: the slope of the look that counts at each moment (1 short, 2 long, 0 none), the span
    over which the lines are followed (the better look's), and how far to both sides."""
    T = 1 + len(x) // STEP; t = np.arange(T) * STEP; L = []
    for n, hop in LOOKS:
        m = hop // STEP; Tl = 1 + len(x) // hop; ix = np.minimum(np.arange(Tl)[:, None] * m + np.arange(m)[None, :], T - 1)
        p, tr, span, reach, e, cnt, md, go, ot = look(x, n, hop, loose, trust, ties, None if speed is None else speed[ix].max(1), log, None if brk is None else brk[ix].any(1), None if cv is None else cv[ix[:, 0]]); tt = np.arange(len(p)) * hop; f = lambda v: np.interp(t, tt, v)
        near = np.minimum(np.round(t / hop).astype(int), len(p) - 1)
        L.append(dict(slope=np.gradient(f(p)), trust=f(tr), span=span[near], reach=f(reach), e=f(e), md=md[near], go=go[near], ot=ot[near]))
    s, l = L
    use = np.where((s['trust'] >= trust) & (s['trust'] >= l['trust'] - EDGE), 1, np.where(l['trust'] >= trust, 2, 0))
    k = 15; vote = ndi.uniform_filter1d((use == 2).astype(float), k, mode='nearest') - ndi.uniform_filter1d((use == 1).astype(float), k, mode='nearest')
    use = np.where((use > 0) & (vote > 0) & (l['trust'] >= trust), 2, np.where((use > 0) & (vote <= 0) & (s['trust'] >= trust), 1, use))
    use = np.where(s['md'] & ~l['md'], 1, np.where(l['md'] & ~s['md'], 2, use))            # a loss one look found the sound again across: that look speaks there
    slope = np.where(use == 1, s['slope'], np.where(use == 2, l['slope'], 0.0))
    return dict(md=s['md'] | l['md'], go=s['go'] | l['go'], ot=s['ot'] | l['ot'], use=use, slope=slope, span=np.maximum(s['span'], l['span']), reach=np.maximum(s['reach'], l['reach']), reach_s=s['reach'], span_s=s['span'], which=np.where(s['span'] >= l['span'], 1, 2) * ((s['span'] > 0) | (l['span'] > 0)), s=s, l=l)

def tau_of(n, p):
    c = np.interp(np.arange(n) / STEP, np.arange(len(p)), p - p.mean()); return np.concatenate([[0.0], np.cumsum(2.0 ** (c / 1200.0))[:-1]])

def flatten(x, p):
    tau = tau_of(len(x), p); m = int(tau[-1]) + 1; pos = np.interp(np.arange(m), tau, np.arange(len(x))); return warp.read_at(x, pos), tau

def straight(q, md):
    """A loss the sound was found again across (md): how far the sound moved in all is known, the path is not, so the path is a straight line."""
    if md is None or not md.any(): return q
    q = q.copy(); T = len(q); idx = np.nonzero(md)[0]
    for r_ in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
        a, b = max(r_[0] - 1, 0), min(r_[-1] + 2, T)            # a moment more on each side: the two looks' moments lie that far apart
        if a == 0 and b == T: continue
        va = q[a - 1] if a > 0 else q[b]; vb = q[b] if b < T else va
        q[a:b] = va + (vb - va) * (np.arange(b - a) + 1.0) / (b - a + 1.0)
    return q

def gathered(x, c):
    """How still the lines of x stand once x is flattened with the curve c (cents per 10 ms, taken as it is): per moment
    of x and per block of the spectrum, how gathered the block's energy is. Seen through the long window: a line that
    stands still has its energy in two or three bins, one that still moves spreads it over more, and two sounds that do
    not lie at the same place hold twice the lines. The measure: the number of values in the block times the sum of the
    squared energies over the square of their sum (1: spread evenly, 2: noise, the number of values: all in one). Blocks
    FLOOR_HZ wide that fade in and out and overlap by half, so that a line moving from one to the next changes nothing."""
    T = len(c); n = len(x); cs = np.interp(np.arange(n) / STEP, np.arange(T), c); tau = np.concatenate([[0.0], np.cumsum(2.0 ** (cs / 1200.0))[:-1]])
    y = warp.read_at(x, np.interp(np.arange(int(tau[-1]) + 1), tau, np.arange(n))); N = LOOKS[1][0]; hop = GHOP; M = FINE * N
    F = frames(y, N, hop); h = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N); Tf = len(F)
    blk = FINE * max(8, int(round(FLOOR_HZ * N / SR))); k0 = int(np.ceil(FMIN * M / SR)); nb = int((FMAX * M / SR - k0) // blk) - 1
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(2 * blk) + 0.5) / (2 * blk)); sw = w.sum(); G = np.ones((Tf, nb))
    for a in range(0, Tf, 64):          # in batches (memory)
        P = np.abs(sfft.rfft(F[a:a + 64] * h, M, axis=1)[:, k0:k0 + (nb + 1) * blk]) ** 2; P = P + DUST * P.mean(1)[:, None]
        for j in range(nb):
            Q = P[:, j * blk:(j + 2) * blk]; s1 = Q @ w; s2 = (Q * Q) @ w; ok = s1 > 0; G[a:a + 64, j] = np.where(ok, sw * s2 / np.where(ok, s1, 1.0) ** 2, 1.0)
    ty = np.clip(tau[np.minimum(np.arange(T) * STEP, n - 1)] / hop, 0, Tf - 1); i0 = np.minimum(ty.astype(np.int64), max(Tf - 2, 0)); fr = (ty - i0)[:, None]
    return G[i0] * (1 - fr) + G[np.minimum(i0 + 1, Tf - 1)] * fr

def stiller(Ga, Gb):
    """How many times stiller the lines stand in Ga than in Gb (gathered, per moment or one row): block by block the ratio,
    over the blocks that hold lines in either, averaged as ratios are (every such block has one voice, the strongest lines
    no more than the weakest); and how many blocks spoke. 1 where no block holds lines: nothing can be said."""
    m = np.maximum(Ga, Gb) > LINES; lr = np.where(m, np.log(np.maximum(Ga, 1.0) / np.maximum(Gb, 1.0)), 0.0)
    return np.exp(lr.sum(-1) / np.maximum(m.sum(-1), 1)), m.sum(-1)

def across(x, out, ea, a, b, eb):
    """A passage [a, b) between two stretches (the one before begins at ea, the one after ends at eb) inside which the curve
    out takes a step: how alike the sound's lines just after the passage are those just before it once the sound is
    flattened with out (0 .. 1), and how much further the curve would have to step for them to lie at the same place
    (cents, looked for ROUND either way). The short look's lines, of up to six moments on each side that lie two moments
    clear of the passage, each side's taken together."""
    T = len(out); fb = np.arange(max(ea, a - 8), max(ea + 1, a - 2)); fa = np.arange(min(eb - 1, b + 2), min(eb, b + 8))
    lo = max(0, fb[0] - 6); hi = min(T - 1, fa[-1] + 6); seg = x[lo * STEP:min(len(x), hi * STEP + 1)]; n = len(seg)
    if n < 2: return 0.0, 0.0
    cur = out[lo:hi + 1] - out[a - 1]; cur[a - lo:b - lo] = 0.0          # inside the passage the curve stands still (the step at its end), and outside the two stretches as well
    if ea > lo: cur[:ea - lo] = cur[ea - lo]
    if eb <= hi: cur[eb - lo:] = cur[eb - 1 - lo]
    cs = np.interp(np.arange(n) / STEP, np.arange(hi - lo + 1), cur); tau = np.concatenate([[0.0], np.cumsum(2.0 ** (cs / 1200.0))[:-1]])
    L = lines(warp.read_at(seg, np.interp(np.arange(int(tau[-1]) + 1), tau, np.arange(n))), LOOKS[0][0], STEP)[0]
    at = lambda t: np.clip(np.round(tau[np.minimum((t - lo) * STEP, n - 1)] / STEP).astype(np.int64), 0, len(L) - 1)
    Pb = L[at(fb)].sum(0); Pa = L[at(fa)].sum(0); den = np.sqrt((Pb ** 2).sum() * (Pa ** 2).sum())
    if den <= 0: return 0.0, 0.0
    S = int(round(ROUND / RES)); C = np.array([(Pb[max(0, -s_):NB - max(0, s_)] * Pa[max(0, s_):NB - max(0, -s_)]).sum() for s_ in range(-S, S + 1)]) / den
    j = int(C.argmax())
    if j == 0 or j == 2 * S: return 0.0, 0.0                    # most alike at the edge of where it is looked for: not found
    l0, l1, l2 = (np.log(max(C[j + q_], 1e-9)) for q_ in (-1, 0, 1)); den = l0 - 2 * l1 + l2; off = 0.5 * (l0 - l2) / min(den, -1e-9) if den < -1e-9 else 0.0
    return float(C[j]), float((j - S + min(max(off, -1.0), 1.0)) * RES)

def compose(q, span, reach=None, reach_s=None, span_s=None, plain=False, runs=None, raw=None, apart=None, goes=None, md=None, other=None, log=None, prior=None, x=None, stepped=None):
    """The chain q (cents, 10 ms grid) as far as it is kept. A stretch: moments in a row whose lines are followed. Inside a
    stretch the chain counts as it is where the lines are followed over a long span; where they are followed over short
    spans only (events handing over to each other can make such a chain run away) it counts against its own average over
    REACH times that span. A loss the sound was found again across (md) is a straight line (see straight).
    Between stretches nothing is read, and nothing may be made up there: the curve stands still, and the next stretch
    goes on where the last one left it. It begins afresh only where a new sound starts. That is at a pause (apart): the
    sounds on both sides stand around their own averages, and the step between them lies inside the pause. And it is at
    a place where another sound may start (other: what follows a loss is most alike what came before it somewhere else,
    or was found there and not proven), for events only: an event (a stretch shorter than LONG, or without lines
    followed over a long span) stands around its own average there, at the level the sustained sounds hold around it
    (zero if there are none), and the curve steps to it just before it begins, over RAMP, or in a straight line where
    the two stretches are less than GAP apart. A sustained sound does not begin afresh there (an engine that drops
    fast looks like another sound starting): it goes on where the last sustained sound left the curve. Before the first
    stretch and after the last the curve stands still as well.
    The curve that is kept (the sound x is given) holds less, so that nothing in it is made up. Beside a sustained sound an
    event is not kept: short stretches next to an engine or a glide are where a reading goes wrong (a line that beats, a
    thump, a few moments in noise), and a step to an event and back is made up wherever the sound in truth runs on (it
    moved faster than it was looked for, or its strongest line handed over to another). Where no sound is sustained the
    events are the sound (zaps, chirps): the longest part stands, and every other part keeps its place only if the step
    to it is proven: with the sound flattened by the curve, the lines after the step are ALIKE those before it (see
    across), and the part is put where they are found, one part after the other going out from the longest. A step
    under NEAR needs no proof. A part whose place is not proven is not kept: the curve stands still across it.
    stepped (a list): the passages [a, b) in which the curve takes a step of NEAR or more are added to it."""
    T = len(q); held = (span > 0).copy(); run = (held if runs is None else (runs | held)).copy(); out = np.zeros(T); fs = SR / STEP
    q = straight(q, md); pr = np.zeros(T) if prior is None else np.asarray(prior, float)       # where nothing is read the curve moves as pr does (without one it stands still)
    # a crumb (a stretch shorter than CRUMB next to a short loss) is no sound of its own: it hangs onto its neighbour, the curve standing still across the loss
    idx = np.nonzero(run)[0]
    if len(idx):
        rs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for k in range(len(rs) - 1):
            a, b = rs[k][-1] + 1, rs[k + 1][0]
            if b - a <= GAP * fs and min(len(rs[k]), len(rs[k + 1])) < CRUMB * fs and not (apart is not None and apart[a:b].any()) and not (other is not None and other[a:b].any()): run[a:b] = True
    if goes is not None: run |= goes
    if other is not None: run &= ~(other & ~held)             # where another sound may start and nothing is followed strictly, nothing hangs together
    idx = np.nonzero(run)[0]
    for r_ in (np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []):         # moments that hang together without one that is followed strictly: nothing is read there, it is no stretch
        if not held[r_[0]:r_[-1] + 1].any(): run[r_[0]:r_[-1] + 1] = False
    while True:
      out[:] = 0.0
      cut = np.zeros(T, bool); cut[:-1] = ~(run[:-1] & run[1:])                  # cut[t]: t and t+1 do not hang together
      idx = np.nonzero(run)[0]
      if len(idx) == 0: return out, held
      seg = np.concatenate([[0], np.cumsum(cut[:-1])]); ends = []; sus = []; drop = None
      for g in np.unique(seg[run]):
          m = np.nonzero((seg == g) & run)[0]; a, b = m[0], m[-1] + 1; qq = q[a:b]; inc = np.diff(qq); hh = held[a:b]; long_ = (span if raw is None else raw)[a:b] >= LONG
          sidx = np.nonzero(~long_ & hh)[0]
          if len(sidx) and not plain:
              for sr_ in np.split(sidx, np.nonzero(np.diff(sidx) > 1)[0] + 1):
                  c, d = sr_[0], sr_[-1] + 1
                  if d - c < 2: continue
                  z = qq[c:d]; cs = np.concatenate([[0.0], np.cumsum(z)]); rr = np.maximum(1, np.round(REACH * span[a + c:a + d] * fs)).astype(int)
                  lo = np.maximum(0, np.arange(d - c) - rr); hi = np.minimum(d - c, np.arange(d - c) + rr + 1); h = z - (cs[hi] - cs[lo]) / (hi - lo)
                  inc[c:d - 1] = np.diff(h)
          nh = ~(hh[:-1] & hh[1:]); inc[nh] = np.diff(pr[a:b])[nh]              # followed loosely only: nothing is read there
          o = np.concatenate([[0.0], np.cumsum(inc)]); out[a:b] = o - o.mean(); ends.append((a, b)); sus.append(bool((long_ & hh).any()) and b - a >= LONG * fs)
      # between the stretches
      K = len(ends); gap = int(round(GAP * fs)); ramp = int(round(RAMP * fs)); up = (np.arange(ramp) + 1.0) / (ramp + 1.0)
      line = lambda va, vb, n_: va + (vb - va) * (np.arange(n_) + 1.0) / (n_ + 1.0)
      kinds = []                                      # what lies between stretch k and the next: 2 a pause, 1 a place where another sound may start, 0 nothing known
      for k in range(K - 1):
          a, b = ends[k][1], ends[k + 1][0]
          kinds.append(2 if (apart is not None and apart[a:b].any()) else 1 if (other is not None and other[a:b].any()) else 0)
      evt = np.zeros(K, bool); k = 0                  # evt: the stretch belongs to a part that holds no sustained sound
      while k < K:
          j = k
          while j < K - 1 and kinds[j] != 2: j += 1                  # the stretches k .. j: no pause between them
          parts = [[k]]                                               # a part: stretches with nothing known between them, each going on where the one before left the curve
          for i in range(k + 1, j + 1):
              if kinds[i - 1] == 0: a, b = ends[i]; e_ = ends[i - 1][1] - 1; out[a:b] += out[e_] + (pr[a] - pr[e_]) - out[a]; parts[-1].append(i)
              else: parts.append([i])
          sp = [any(sus[i] for i in P) for P in parts]; S = [P for P, s_ in zip(parts, sp) if s_]
          for u in range(1, len(S)):                                  # the parts that hold a sustained sound: each goes on where the one before left the curve
              e_ = ends[S[u - 1][-1]][1] - 1; a = ends[S[u][0]][0]; d = out[e_] + (pr[a] - pr[e_]) - out[a]
              for i in S[u]: out[ends[i][0]:ends[i][1]] += d
          if S:                                                       # and together they stand around the average of their sustained sounds
              C = [i for P in S for i in P if sus[i]]; m = sum(out[ends[i][0]:ends[i][1]].sum() for i in C) / sum(ends[i][1] - ends[i][0] for i in C)
              for P in S:
                  for i in P: out[ends[i][0]:ends[i][1]] -= m
          for P, s_ in zip(parts, sp):                                # a part of events only: around its own average, at the level the sustained sounds hold around it
              if s_: continue
              m = sum(out[ends[i][0]:ends[i][1]].sum() for i in P) / sum(ends[i][1] - ends[i][0] for i in P)
              bf = [R for R in S if R[0] < P[0]]; af = [R for R in S if R[0] > P[0]]
              B = out[ends[bf[-1][-1]][1] - 1] if bf else (out[ends[af[0][0]][0]] if af else 0.0)
              for i in P: out[ends[i][0]:ends[i][1]] += B - m; evt[i] = True
          if x is not None and drop is None and S and not all(sus[i] for i in range(k, j + 1)):         # beside a sustained sound an event is not kept
              drop = ([i for i in range(k, j + 1) if not sus[i]], None)
          if x is not None and drop is None and len(parts) > 1 and not S:                                # events only: placed against the longest part, each by the step to it (those ahead of it by the step from them)
              f0 = max(range(len(parts)), key=lambda u_: sum(ends[i][1] - ends[i][0] for i in parts[u_]))
              for u in list(range(f0 - 1, -1, -1)) + list(range(f0 + 1, len(parts))):
                  P = parts[u]; back = u < f0; i0_, i1_ = (P[-1], P[-1] + 1) if back else (P[0] - 1, P[0])
                  a, b = ends[i0_][1], ends[i1_][0]; d = out[b] - out[a - 1]
                  if abs(d - (pr[b] - pr[a - 1])) < NEAR: continue
                  lk, more = across(x, out, ends[i0_][0], a, b, ends[i1_][1])
                  if lk < ALIKE: drop = (P, (d, lk, more, a, b)); break
                  if log: log('   a step of %+.0f cents between %.2f and %.2f s: the two sides are alike %.2f there, %+.0f cents off: placed' % (d, a / fs, b / fs, lk, more))
                  for i in P: out[ends[i][0]:ends[i][1]] += -more if back else more
          k = j + 1
      if drop is None: break
      for i in drop[0]:
          if log and drop[1] is None: log('   stretch %.2f to %.2f s: an event beside a sustained sound, not kept' % (ends[i][0] / fs, ends[i][1] / fs))
          elif log: log('   stretch %.2f to %.2f s: an event whose place is not proven, not kept (a step of %+.0f cents between %.2f and %.2f s: the two sides are alike %.2f there, %+.0f cents off)' % (ends[i][0] / fs, ends[i][1] / fs, drop[1][0], drop[1][3] / fs, drop[1][4] / fs, drop[1][1], drop[1][2]))
          run[ends[i][0]:ends[i][1]] = False; held[ends[i][0]:ends[i][1]] = False
    if log:
        for i in range(K):
            log('   stretch %.2f to %.2f s, %s, %+.0f to %+.0f cents%s' % (ends[i][0] / fs, ends[i][1] / fs, 'sustained' if sus[i] else 'an event of its own' if evt[i] else 'an event', out[ends[i][0]], out[ends[i][1] - 1],
                '' if i == K - 1 else '; then ' + ('a pause', 'another sound may start', 'nothing known')[2 - kinds[i]]))
    for p in range(K + 1):
        a = ends[p - 1][1] if p > 0 else 0; b = ends[p][0] if p < K else T          # the passage [a, b)
        if b <= a: continue
        n_ = b - a
        if p == 0: out[a:b] = out[b] + pr[a:b] - pr[b]
        elif p == K: out[a:b] = out[a - 1] + pr[a:b] - pr[a - 1]
        else:
            va, vb = out[a - 1], out[b]; step = np.zeros(n_)        # the passage moves as pr does; what that leaves of the way from va to vb is a step, taken where step goes from 0 to 1
            if kinds[p - 1] == 2:                                   # inside the pause
                ix = np.nonzero(apart[a:b])[0]; p0, p1 = ix[0], ix[-1] + 1; step[p1:] = 1.0; step[p0:p1] = line(0.0, 1.0, p1 - p0)
            elif n_ <= gap: step = line(0.0, 1.0, n_)
            elif evt[p]: step[n_ - ramp:] = up                      # events begin afresh there: the curve steps to them just before they begin
            else: step[:ramp] = up; step[ramp:] = 1.0               # after events, at once back to where the sustained sound goes on
            out[a:b] = va + (pr[a:b] - pr[a - 1]) + ((vb - va) - (pr[b] - pr[a - 1])) * step
            if stepped is not None and kinds[p - 1] != 2 and abs((vb - va) - (pr[b] - pr[a - 1])) >= NEAR: stepped.append((a, b))
    return out, held

def slow_part(v):
    """The slow part of a curve: smoothed over SLOW moments, the curve carried on past both ends as it arrives there (mirrored
    through its end points), so that a steady glide has no fast part at the ends of the file either."""
    T = len(v); R = int(4.0 * SLOW + 0.5)
    if T < 2: return np.array(v, float)
    i = np.arange(1, R + 1); left = 2.0 * v[0] - v[np.minimum(i, T - 1)][::-1]; right = 2.0 * v[-1] - v[np.maximum(T - 1 - i, 0)]
    return ndi.gaussian_filter1d(np.concatenate([left, v, right]), SLOW, mode='nearest')[R:R + T]

def proven(x, out, log=None):
    """The curve with its fast part only where that part is proven: where the lines of x stand stiller once x is flattened
    with the whole curve than with the curve's slow part alone. Two sounds close to each other in pitch beat, their lines
    flutter, and a curve that follows the flutter adds one; a reading that jitters adds its jitter. Also: how much of the
    fast part is kept at each moment (0 .. 1), and how many times stiller the lines stand with it."""
    T = len(out); fs = SR / STEP; slow = slow_part(out); fast = out - slow; keep = np.ones(T); ratio = np.ones(T)
    if np.abs(fast).max() > 1.0:
        w = 2 * int(round(0.15 * fs)) + 1; m_ = out.mean()
        ratio = stiller(ndi.uniform_filter1d(gathered(x, out - m_), w, axis=0, mode='nearest'), ndi.uniform_filter1d(gathered(x, slow - m_), w, axis=0, mode='nearest'))[0]
        keep = np.clip((ratio - FASTKEEP[0]) / (FASTKEEP[1] - FASTKEEP[0]), 0.0, 1.0)
        keep = ndi.gaussian_filter1d(keep, 5.0, mode='nearest'); out = slow + keep * fast
        if log: log('the fast part of the curve (rms %.1f cents): kept %.2f on average, least %.2f' % (fast.std(), keep.mean(), keep.min()))
    return out, keep, ratio

def bound(w, lin=1200.0, top=1800.0):
    """The curve a sound is flattened with stays inside +-top cents (as it is up to +-lin)."""
    a = np.abs(w); return np.where(a <= lin, w, np.sign(w) * (lin + (top - lin) * np.tanh((a - lin) / (top - lin))))

def cuts(x, power=False):
    """Where a sound starts or stops abruptly, the ends of the file included: (sample, +1) the sound starts with that sample,
    (sample, -1) it stopped with the sample before. A window that lies across such a cut holds a tone that stops, and the
    frequency its phase says it has is then off by up to 0.4 bins (5 Hz with the short window, 1 Hz with the long one)
    whatever the tone's own frequency: nothing for a line at 4 kHz, a semitone and more for one at 60 Hz. Read like that,
    the curve of a low sound runs away at every cut. Recordings cut in two, against the same recordings read whole: off by
    50 cents on average over the 40 ms next to the cut, up to 180, where the sound itself hardly moves; with silence around
    the piece the curve runs on into the silence. A sound whose lines lie high reads right up to the cut.
    power: x holds the sound's power sample by sample (its squares), not the sound."""
    n = len(x); B = -(-n // STEP); out = []
    if B < 1: return out
    sq = np.zeros(B * STEP); sq[:n] = np.asarray(x, float) if power else np.asarray(x, float) ** 2; E = np.zeros(B + 6)         # the level of every hundredth of a second: block b at b + 3, silence outside the file
    E[3:3 + B] = sq.reshape(B, STEP).sum(1) / np.minimum(n - np.arange(B) * STEP, STEP)
    lev = lambda b: E[b + 3]
    quiet_after = lambda b: lev(b) < JUMP * max(lev(b + 1), lev(b + 2)); quiet_before = lambda b: lev(b) < JUMP * max(lev(b - 1), lev(b - 2))
    for b in range(B + 1):
        if b < B and quiet_after(b - 1) and not quiet_after(b):         # a sound starts in block b: at its first sample that is not quiet
            ref = JUMP * max(lev(b), lev(b + 1)); at = b * STEP; hit = np.nonzero(sq[at:min(n, (b + 2) * STEP)] >= ref)[0]
            out.append((at + int(hit[0]) if len(hit) else at, 1))
        if b >= 1 and quiet_before(b) and not quiet_before(b - 1):       # a sound stops in block b - 1: after its last sample that is not quiet
            ref = JUMP * max(lev(b - 1), lev(b - 2)); at = min(n, b * STEP); lo = max(0, (b - 2) * STEP); hit = np.nonzero(sq[lo:at] >= ref)[0]
            out.append((lo + int(hit[-1]) + 1 if len(hit) else at, -1))
    return out

def beside_cuts(found, T):
    """Per frame: the frame whose reading counts for it (itself where no cut is near), -1 = not followed (on the quiet side
    of a cut, or beside a cut in a sound too short to have a frame that is left alone)."""
    src = np.arange(T); spoiled = np.zeros(T, bool); dead = np.zeros(T, bool); zones = []     # zones: the frames a .. b beside a cut, and the frame they hold on to
    for k, (p, d) in enumerate(found):
        if d > 0:
            a = (p - 1) // STEP + 1; h = (p + NEARCUT - 1) // STEP + 1; zones.append((a, h - 1, h))           # the frames from p to just under p + NEARCUT
            back = max(found[k - 1][0] if k > 0 else p - REACHCUT, p - REACHCUT); t = min(a - 1, T - 1)
            while t >= 0 and t * STEP >= back: dead[t] = True; t -= 1
        else:
            b = (p - 1) // STEP; h = (p - NEARCUT) // STEP; zones.append((h + 1, b, h))                         # the frames from just over p - NEARCUT to just under p
            on = min(found[k + 1][0] - 1 if k + 1 < len(found) else p + REACHCUT, p + REACHCUT); t = max(b + 1, 0)
            while t < T and t * STEP <= on: dead[t] = True; t += 1
    for a, b, h in zones: spoiled[max(0, a):min(T - 1, b) + 1] = True
    for a, b, h in zones:
        ok = 0 <= h < T and not spoiled[h] and not dead[h]
        for t in range(max(0, a), min(T - 1, b) + 1):
            if ok: src[t] = h
            else: dead[t] = True
    src[dead] = -1
    return src

def lowness(x, c):
    """How low the lines of the frame around sample c lie: the mean of 1 / frequency over them (per Hz), each line as strong
    as it is drawn. Times 1731 and a shift in Hz: how many cents that shift moves the lines on average. No lines: a large number."""
    lead = 10; seg = np.zeros(2 * lead * STEP); a = c - lead * STEP; lo = max(0, a); hi = min(len(x), a + len(seg))
    if hi > lo: seg[lo - a:hi - a] = x[lo:hi]
    L = lines(seg, LOOKS[0][0], STEP)[0]
    if len(L) <= lead: return 1e9
    row = L[lead] ** 2; den = row.sum()
    return float((row / (FMIN * 2.0 ** (np.arange(NB) * RES / 1200.0))).sum() / den) if den > 0 else 1e9

def track(x, detail=False, log=None):
    """The pitch movement of x (mono) in cents, a value every 10 ms; which look follows each moment (1 short, 2 long, 0 none);
    the frames' energies. Rounds: the sound is flattened with the curve read so far and read again, loosely, until nothing
    more is left to read (the curve it is flattened with: the chain as far as it hangs together, moving as the curve before
    did where nothing is read, its fast part only where that is proven); then once more, strictly. What is kept: see compose."""
    n = len(x); T = 1 + n // STEP; fs = SR / STEP; c = np.zeros(T); info = []; y, tau = x, np.arange(n, dtype=float); it = 0; before = None; stall = 0; kept = []
    made = np.zeros(T, bool)        # the moments at which the curve the sound is flattened with takes a step that nothing proves: no reading follows them
    def steps_of(st): m = np.zeros(T, bool); [m.__setitem__(slice(a, b), True) for a, b in st]; return m
    e = (frames(x, 4096, STEP) ** 2).mean(1); t_ = np.arange(T)
    k = int(round(0.5 * fs)) // 2; cm = ndi.maximum_filter1d(e, 2 * k + 1, mode='constant')
    brk = e < PAUSE * np.minimum(cm[np.clip(t_ - k, 0, T - 1)], cm[np.clip(t_ + k, 0, T - 1)])       # a pause: far under what sounds within half a second before and after it
    def read(y, tau, c, loose, trust):
        ty = tau[np.minimum(t_ * STEP, n - 1)] / STEP                  # where x's frames are in the flattened sound, in its frames
        k = 2.0 ** (-(c - c.mean()) / 1200.0)                          # a span of the flattened sound's time, in the sound's own time
        sp = np.abs(np.gradient(ndi.uniform_filter1d(c, 5, mode='nearest'))) * fs; sp = ndi.maximum_filter1d(sp, 31, mode='nearest') * k      # how fast the curve moves around each moment, cents per second of the flattened sound
        Ty = 1 + len(y) // STEP; own = np.interp(np.arange(Ty), ty, t_); sp_y = np.interp(own, t_, sp)
        by = brk[np.clip(np.round(own).astype(int), 0, T - 1)]; by[np.clip(np.round(ty[brk]).astype(int), 0, Ty - 1)] = True
        B = both(y, loose=loose, trust=trust, speed=sp_y, log=log, brk=by, cv=np.interp(own, t_, c))
        at = lambda v: np.interp(ty, np.arange(len(v)), v); near = np.clip(np.round(ty).astype(int), 0, len(B['use']) - 1); z = lambda v: np.where(brk, 0.0, v)
        r = at(np.cumsum(B['slope'])); F = dict(span=z(B['span'][near] * k), raw=z(B['span'][near]), md=B['md'][near], go=B['go'][near] & ~brk, ot=B['ot'][near], reach=z(at(B['reach']) * k), reach_s=z(at(B['reach_s']) * k), span_s=z(B['span_s'][near] * k), which=B['which'][near])
        # a sound found again across a loss (a sound's lines fit those of its own harmonics in part, and can be found where
        # they are not): the curve goes through the loss only if the sound is heard to move like that inside the loss, that is
        # if its lines stand stiller there with the curve gliding through than with a curve that takes the same step at once
        # in the middle of the loss (the two curves are the same outside the loss). Not proven: nothing is known of the passage.
        # A finding that moves the curve by less than NEAR cents in all needs no proof: it cannot add more than that, and a curve
        # that stood still at every such place would read too little every time (a glide under a row of clicks).
        for key in ('span', 'raw', 'reach', 'reach_s', 'span_s'): F[key] = np.where(made, 0.0, F[key])
        F['md0'] = F['md'].copy(); idx = np.nonzero(F['md'])[0]; W = int(round(AROUND * fs)); qs = straight(c + r, F['md'])
        for run in (np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []):
            a, b = max(run[0] - 1, 0), min(run[-1] + 2, T)
            if a == 0 or b == T: continue
            lo, hi = max(0, a - W), min(T, b + W); va, vb = qs[a - 1], qs[b]; qa = qs[lo:hi] - 0.5 * (va + vb)
            qb = qa.copy(); h_ = (a + b) // 2; qb[a - lo:h_ - lo] = va - 0.5 * (va + vb); qb[h_ - lo:b - lo] = vb - 0.5 * (va + vb)
            if abs(vb - va) < NEAR:
                if log: log('   found again %.2f to %.2f s, %.0f cents: kept as it is' % (run[0] / fs, (run[-1] + 1) / fs, vb - va))
                continue
            seg = x[lo * STEP:min(n, hi * STEP)]; k0, k1 = max(a - lo - 2, 0), min(b - lo + 2, hi - lo)
            ratio, said = stiller(gathered(seg, qa)[k0:k1].mean(0), gathered(seg, qb)[k0:k1].mean(0)); stays = ratio >= STEADY
            if log: log('   found again %.2f to %.2f s, %.0f cents: the lines stand %.2f times stiller with the glide than with the step (%d blocks): %s' % (run[0] / fs, (run[-1] + 1) / fs, vb - va, ratio, said, 'kept' if stays else 'not proven, nothing is known of this passage'))
            if not stays:
                m = slice(run[0], run[-1] + 1)
                for key in ('span', 'raw', 'reach', 'reach_s', 'span_s'): F[key][m] = 0.0
                F['md'][m] = False; F['ot'] = F['ot'].copy(); F['ot'][m] = True       # found somewhere else and not proven: another sound may start there
        return r, F
    while True:
        r, F = read(y, tau, c, 3.0, 0.6); q = c + r; got = F['span'] > 0
        if not got.any():
            if log: log('round %d: nothing followed' % it)
            break
        seg = np.cumsum(np.concatenate([[0], np.diff(got.astype(int)) != 0])); rr = r.copy()
        for g in np.unique(seg[got]): m = seg == g; rr[m] -= rr[m].mean()
        left = float(np.sqrt((rr[got] ** 2).mean())); most = float(np.abs(rr[got]).max()); info.append(left)
        if log: log('round %d: followed %.2f, left to read rms %.1f, at most %.0f' % (it, got.mean(), left, most))
        it += 1; stall = stall + 1 if (before is not None and left > 0.7 * before) else 0; kept.append((left, float(got.mean()), q, F, got, c))
        if (left < DONE[0] and most < DONE[1]) or stall >= 2 or it >= PASSES: break
        st = []; before = left; c = bound(proven(x, compose(q, F['span'], plain=True, apart=brk, goes=F['go'], md=F['md'], other=F['ot'], prior=c, stepped=st)[0], log)[0]); y, tau = flatten(x, c); made = steps_of(st)
    runs = None; dbg = {}
    if len(kept) > 1:               # the reading that left the least to read (of those that follow about as much as the best) is the one to go on from
        top = max(k[1] for k in kept); best = min((k for k in kept[1:] if k[1] >= 0.8 * top), key=lambda k: k[0], default=kept[-1])
        if best is not kept[-1] and best[0] < 0.7 * kept[-1][0]:
            if log: log('going on from the round that left %.1f' % best[0])
            q, F, got = best[2], best[3], best[4]; c = None
    if got.any():
        st = []; g = bound(proven(x, compose(q, F['span'], plain=True, apart=brk, goes=F['go'], md=F['md'], other=F['ot'], prior=kept[-1][5] if c is not None else best[5], stepped=st)[0], log)[0]); runs = got | F['go']
        if c is None or np.abs(g - c)[got].max() > 2.0: c = g; y, tau = flatten(x, c); made = steps_of(st)
        r, F = read(y, tau, c, 1.0, TRUST); q = c + r
        # what reading the flattened sound again still finds: where that is no less than half of the curve's own movement, the curve is not kept
        # (by its slow part: whether its fast part counts is for the proof to say)
        w = 2 * int(round(0.5 * fs)) + 1
        def loc(v): m = ndi.uniform_filter1d(v, w, mode='nearest'); return np.sqrt(np.maximum(ndi.uniform_filter1d(v * v, w, mode='nearest') - m * m, 0.0))
        rg = np.concatenate([[0.0], np.cumsum(np.where(F['md0'][1:] | F['md0'][:-1], 0.0, np.diff(r)))])        # without the steps where a loss was mended in this reading
        rg = slow_part(rg)
        unsure = (loc(rg) > SURE[0]) & (loc(rg) > SURE[1] * loc(q)) & (F['span'] > 0)
        hd = F['span'] > 0
        if hd.any():                # and as a whole: a reading that leaves half of its own movement to be read again is none
            sg = np.cumsum(np.concatenate([[0], np.diff(hd.astype(int)) != 0])); ra = rg.copy(); qa = q.copy()
            for g_ in np.unique(sg[hd]): m_ = sg == g_; ra[m_] -= ra[m_].mean(); qa[m_] -= qa[m_].mean()
            if ra[hd].std() > 10.0 and ra[hd].std() > SURE[1] * qa[hd].std(): unsure = hd.copy()
        if log: log('last: followed %.2f, read on top rms %.1f, not kept %.2f' % ((F['span'] > 0).mean(), r[F['span'] > 0].std() if (F['span'] > 0).any() else 0, unsure.mean()))
        dbg = dict(r=r.copy(), c=c.copy(), rg=rg, unsure=unsure, hd=hd)
        for key in ('span', 'raw', 'reach', 'reach_s', 'span_s'): F[key] = np.where(unsure, 0.0, F[key])
    # beside a cut (see cuts) the reading counts only where the sound's lines lie high enough for the look that reads there;
    # where they do not, the curve stands still (it takes the reading of the nearest frame the cut leaves alone), which loses
    # the movement of those 40 ms where it was read right and never adds one. On the quiet side of a cut nothing is followed.
    src = beside_cuts(cuts(x), T); q = q.copy(); F = {key: np.array(v) for key, v in F.items()}; runs = None if runs is None else runs.copy(); t = 0
    while t < T:
        s = int(src[t])
        if s < 0:
            for key in ('span', 'raw', 'reach', 'reach_s', 'span_s'): F[key][t] = 0.0
            F['go'][t] = False
            if runs is not None: runs[t] = False
            t += 1; continue
        if s == t: t += 1; continue
        e_ = t                      # the frames t .. e_ - 1 hold on to frame s. Going out from s their own reading counts as far as the look
        while e_ < T and src[e_] == s and e_ != s: e_ += 1      # that follows s follows them too, if the sound's lines lie high enough for it
        look = int(F['which'][s]) if F['span'][s] > 0 else 0
        high = look in (1, 2) and (1200.0 / np.log(2.0)) * SHIFT[look - 1] * lowness(x, s * STEP) < STAND
        frm = s
        for u in (range(e_ - 1, t - 1, -1) if t < s else range(t, e_)):
            if high and frm == (u + 1 if t < s else u - 1) and F['which'][u] == look and F['span'][u] > 0: frm = u; continue
            q[u] = q[frm]
            for key in F: F[key][u] = F[key][frm]
            if runs is not None: runs[u] = runs[frm]
        t = e_
    out, held = compose(q, F['span'], F['reach'], F['reach_s'], F['span_s'], runs=runs, raw=F['raw'], apart=brk, goes=F['go'], md=F['md'], other=F['ot'], log=log, x=x)
    out = ndi.gaussian_filter1d(out, 1.0, mode='nearest')
    pre = out; out, keep, ratio = proven(x, out, log)         # the fast part of the curve has to earn its place (see proven)
    held = np.where(held, F['which'], 0)
    return (out, held, e, dict(q=q, span=F['span'], reach=F['reach'], info=info, F=F, runs=runs, brk=brk, pre=pre, keep=keep, ratio=ratio, dbg=dbg)) if detail else (out, held, e)
