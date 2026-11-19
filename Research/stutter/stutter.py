# The stutter engine (STUTTER; reference for Source/Core/Stutter.h): a slice of the sound played several times over,
# in place of what came after it.
#   wet[s0 + j L + i] = wet[s0 + i],  j = 1 .. reps - 1,  i = 0 .. L - 1          (one event: start s0, slice L, reps)
# Read off blind: the places where the sound is, sample for sample, what it was L samples before, for a stretch at
# least as long as L. They are looked for with a table of all windows of 64 samples (two windows with the same samples
# land in the same place of the table), then each stretch is walked out to its first and last equal sample.
# The map of events (where, how long a slice, how many times) is what is put onto a second sound.
# A stutter cannot be taken out: what the repeats cover is gone. Left of the middle the repeats are cut out and the
# sound closes up (it gets shorter), which is the nearest thing there is.
import numpy as np
SR = 48000
WIN = 64                         # samples of a window in the table
LMIN, LMAX = int(0.020 * SR), int(2.0 * SR)      # slices looked for: 20 ms to 2 s
EXTRA = int(0.020 * SR)          # an event's repeats have to cover this much at least
LOOP = int(0.3 * SR)             # a slice longer than this may be the sound's own way (a loop that repeats, with breaks in it) and not an event: see own_way()
OWN_OUT = 0.10                   # ... it is, if this share of the sound repeats at that distance outside the stretches themselves
TOL = 2.0 ** -18                 # two samples are "the same" if they are this close (a 24-bit file with dither still passes)
FADE = 96                        # samples of the fade at each end of a slice when the found slices start and end at nothing (2 ms)
KEEP = 0.5                       # a stretch counts if it is at least this share of its slice long (an event: one and a half plays or more)

def table(y):
    """For every sample t the nearest earlier place where the 64 samples from t on occurred before (0: nowhere)."""
    n = len(y); q = np.round(y * 65536.0).astype(np.int64); m = q.shape[1]
    if n < 2 * WIN: return np.zeros(n, np.int64)
    a = np.uint64(0x9E3779B97F4A7C15); mix = np.zeros(n, np.uint64)
    for ch in range(m): mix = mix * np.uint64(0x100000001B3) + q[:, ch].astype(np.uint64)
    # a rolling hash over WIN samples: h[t] = sum mix[t + i] a^(WIN - 1 - i), by prefix sums in whole-number arithmetic that wraps around
    with np.errstate(over='ignore'):
        pw = np.cumprod(np.full(n + 1, a, np.uint64)); inv = np.uint64(pow(int(a), -1, 1 << 64)); ipw = np.cumprod(np.full(n + 1, inv, np.uint64))       # a^(i + 1) and a^-(i + 1)
        pre = np.concatenate([[np.uint64(0)], np.cumsum(mix * ipw[:n])]); h = (pre[WIN:] - pre[:-WIN]) * pw[:n - WIN + 1]           # the same for equal windows wherever they are
    nw = n - WIN + 1; h = h[:nw]
    flat = np.ones(nw, bool); d = (np.diff(q, axis=0) != 0).any(1); c = np.concatenate([[0], np.cumsum(d)]); flat = (c[WIN - 1:WIN - 1 + nw] - c[:nw]) == 0        # a window of one value over and over (silence): says nothing
    order = np.argsort(h, kind='stable'); hs = h[order]; prev = np.zeros(n, np.int64); same = (hs[1:] == hs[:-1])
    prev[order[1:][same]] = order[:-1][same] + 1                       # + 1: 0 stays "nowhere"
    prev[:nw][flat] = 0; return prev

def stretches(y, prev, quiet):
    """The stretches where the sound is what it was L samples before: list of (lo, hi, L) in terms of the earlier
    sample: samples lo + L .. hi + L - 1 equal samples lo .. hi - 1. Silence equals silence, which proves nothing: a
    stretch is cut where a whole slice of it is silent, and it starts and ends no further out than one slice from its
    first and last live sample. And a sound that stands still (a steady offset, a level that moves by less than can be
    told apart) proves nothing by being equal either: such a stretch counts only if its slice length is that of a
    stretch that moves."""
    n = len(y); t = np.where(prev > 0)[0]; out = []; weak = []
    if not len(t): return out
    lag = t - (prev[t] - 1); ok = (lag >= LMIN) & (lag <= LMAX); t, lag = t[ok], lag[ok]
    if not len(t): return out
    cut = np.where((np.diff(t) > WIN) | (np.diff(lag) != 0))[0] + 1; starts = np.concatenate([[0], cut]); ends = np.concatenate([cut, [len(t)]]); seen = {}
    mv = np.concatenate([[0], np.cumsum((np.abs(np.diff(y, axis=0)) > TOL).any(1))])          # mv[b] - mv[a]: how many of the samples a + 1 .. b differ from the one before (a sound that stands still, or moves by less than can be told apart, proves nothing by being equal)
    for s, e in zip(starts, ends):
        L = int(lag[s]); i0 = int(t[(s + e) // 2]) - L                   # from the middle of the run of windows (the table rounds more coarsely than the test below: a run's first and last windows may reach past the stretch)
        if i0 < seen.get(L, -1): continue
        eq = (np.abs(y[L:] - y[:-L]) <= TOL).all(1)                      # eq[i]: sample i + L is what sample i was
        if not eq[i0]: continue
        lo = i0
        while lo > 0 and eq[lo - 1]: lo -= 1
        hi = i0
        while hi < len(eq) and eq[hi]: hi += 1
        seen[L] = hi; lv = np.where(~quiet[lo + L:hi + L])[0] + lo       # the live ones among them
        if not len(lv): continue
        gaps = np.where(np.diff(lv) > L)[0]; first = np.concatenate([[lv[0]], lv[gaps + 1]]); last = np.concatenate([lv[gaps], [lv[-1]]])
        for k, (fl, ll) in enumerate(zip(first, last)):
            a0 = max(lo if k == 0 else int(last[k - 1]) + 1, int(fl) - L + 1); b0 = min(hi if k == len(first) - 1 else int(first[k + 1]), int(ll) + L)
            live = int((~quiet[int(fl) + L:int(ll) + L + 1]).sum()); moving = int(mv[min(int(ll) + L, n - 1)] - mv[int(fl) + L]); need = max(0.2 * min(L, b0 - a0), 4 * WIN)
            if (b0 - a0 >= KEEP * L or (b0 + L == n and b0 - a0 >= 0.25 * L)) and b0 - a0 >= EXTRA and live >= 4 * WIN: (out if min(live, moving) >= need else weak).append((int(a0), int(b0), L))     # cut off by the end of the sound: a quarter of a slice is enough
    have = set(q[2] for q in out); return out + [q for q in weak if q[2] in have]       # a stretch that stands still, or with little sound in it, counts only beside one of the same slice length that moves

def own_way(y, S, quiet):
    """The distances at which the sound repeats of its own accord (a loop), which are not events: (1) one stretch
    that runs from the start of the sound to its end, whatever its length; and for distances over 0.3 s: (2) two
    stretches of which the second goes on with what the first repeats (a loop with a break in it); (3) the sound
    repeating at that distance in many places outside the stretches (a loop with many breaks)."""
    n = len(y); lv = ~quiet; w = np.where(lv)[0]; res = set()
    if not len(w): return res
    f0, f1 = int(w[0]), int(w[-1]); span = max(f1 - f0, 1); nl = len(w)
    for L in sorted(set(q[2] for q in S)):
        P = [q for q in S if q[2] == L]
        if any(a <= f0 + 0.05 * span and b + L >= f1 - 0.10 * span for a, b, _ in P): res.add(L); continue
        if L <= LOOP: continue
        hit = False
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                a1, a2 = P[i][0], P[j][0]; D = L * ((a2 - a1) // L)
                if D <= 0: continue
                e = min(a2 + L, n); m = lv[a2:e]
                if m.sum() and float(((np.abs(y[a2:e] - y[a2 - D:e - D]) <= TOL).all(1) & m).sum()) >= 0.9 * m.sum(): hit = True; break
            if hit: break
        if hit: res.add(L); continue
        eq = (np.abs(y[L:] - y[:-L]) <= TOL).all(1) & lv[L:]
        for a, b, _ in P: eq[a:b] = False
        if float(eq.sum()) >= OWN_OUT * nl: res.add(L)
    return res

def tone(y, s0, L, end):
    """Whether what looks like a slice played again is a steady wave (a held note of a machine, a test tone): the
    slice is itself the same thing over and over at a much shorter distance, and nothing happens where one play ends
    and the next begins. A slice of such a note that was really played again shows a jump (or a fade) there."""
    n = len(y); P0 = y[s0:s0 + L]; P0 = P0 - P0.mean(0); m = P0.shape[1]; nf = 1 << int(np.ceil(np.log2(2 * L)))       # (without its mean: a steady offset is not a wave)
    ac = np.fft.irfft((np.abs(np.fft.rfft(P0, nf, axis=0)) ** 2).sum(1), nf)[:L // 2 + 1]
    c2 = np.concatenate([[0.0], np.cumsum((P0 ** 2).sum(1))]); p = np.arange(L // 2 + 1); r = ac / np.sqrt(np.maximum(c2[L - p] * (c2[L] - c2[p]), 1e-60)); r[:8] = 0.0
    low = np.where(r[8:] < 0.5)[0]                                       # (a slow wave is like itself a few samples on: that is not a period. The distance looked for is one at which the wave comes back after having gone away)
    if not len(low): return False
    r[:8 + int(low[0])] = 0.0; ps = int(np.argmax(r))
    if r[ps] < 0.999: return False
    W = max(ps, 128); seams = 0; stand = 0
    for q in range(s0 + L, end, L):
        if q + W > min(end, n) or q - W < s0: break                                                  # (the sound after the event's end is not a seam)
        ds = float(((y[q - W + ps:q + W] - y[q - W:q + W - ps]) ** 2).sum()) / (2 * W - ps)        # around the seam
        di = float(((y[q - L + ps:q - W] - y[q - L:q - W - ps]) ** 2).sum()) / max(L - W - ps, 1)   # inside the play before it
        seams += 1; stand += ds > 16.0 * di + m * TOL ** 2
    return seams > 0 and 4 * stand <= seams

def goes_on(y, s0, L, end):
    """Whether the sound before the slice or after the last play is the slice once more at another loudness (a held
    note of an instrument that is itself a short loop, fading in or out; a loop through an effect that has to settle).
    A slice that was played again is a cut: the sound before and after it is another. The 100 ms before and after
    are held against the samples one slice length on (or back), with a loudness that may change evenly; alike if
    what is left over is under a hundredth of the power."""
    n = len(y); floor = (16 * TOL) ** 2
    def alike(a, b):
        if len(a) < EXTRA: return False
        a = a - a.mean(0); b = b - b.mean(0)                             # (a steady offset is not a sound that goes on)
        ea = float((a ** 2).sum()); eb = float((b ** 2).sum())
        if ea <= floor * a.size or eb <= floor * b.size: return False
        t = np.linspace(-1.0, 1.0, len(a))[:, None]; B = np.stack([b.ravel(), (b * t).ravel()], 1); g, *_ = np.linalg.lstsq(B, a.ravel(), rcond=None); r = a.ravel() - B @ g
        return float((r ** 2).sum()) < 0.01 * ea and g[0] > 0
    w = min(L, int(0.1 * SR), n - end)
    if w > 0 and alike(y[end:end + w], y[end - L:end - L + w]): return 'the sound goes on the same way after it'
    w = min(L, int(0.1 * SR), s0)
    if w > 0 and alike(y[s0 - w:s0], y[s0 - w + L:s0 + L]): return 'the sound went the same way before it'
    return ''

def apart(y, s0, L, end):
    """Whether every play stands by itself, with silence before and after it (5 ms each, 60 dB under the slice's
    loudest sample): then it is the same note or stroke played again, as a player or a machine does, and not a slice
    cut out of a sound and repeated."""
    g = int(0.005 * SR); thr = max(16 * TOL, 1e-3 * float(np.abs(y[s0:s0 + L]).max())); seams = list(range(s0 + L, end, L))
    return len(seams) > 0 and all(float(np.abs(y[max(q - g, 0):q + g]).max()) <= thr for q in seams)

def dull_ones(y, quiet):
    """The samples that say nothing by being equal to the one a slice before: silence, and a sound that stands still
    (a steady offset: the sample is what its neighbour is)."""
    d = (np.abs(np.diff(y, axis=0)) <= TOL).all(1); dull = quiet.copy(); dull[1:] |= d; dull[:-1] |= d; return dull
def edges(dull, lo, hi, L):
    """How far the start of a stretch may lie after lo, and its end before hi: the samples there that say nothing on
    both sides (they are equal whatever was played)."""
    n = len(dull); a = lo
    while a < hi - 1 and a + L < n and dull[a] and dull[a + L]: a += 1
    b = hi
    while b - 1 > a and dull[b - 1] and dull[b - 1 + L]: b -= 1
    return a, b
def ends_at(s0, L, hi, hi2):
    """Where an event that starts at s0 ends: no later than the stretch (hi + L), no earlier than its last sample
    that says something (hi2 + L). Whole plays where the samples allow it, and as few as they allow."""
    m = int(np.ceil((hi2 + L - s0) / L))
    if m >= 2: return s0 + m * L if s0 + m * L <= hi + L else hi + L
    m = int(np.floor((hi + L - s0) / L + 0.5)); return s0 + m * L if hi + L - L // 2 <= s0 + m * L <= hi + L else hi + L       # nothing in it says anything after the first play: the nearest whole number
def find(y):
    """dict(found, events: list of (s0, L, end): the slice s0 .. s0 + L is played again and again up to sample end;
    fades: whether the plays start and end at nothing; share: how much of the sound the repeats cover; own: the
    distances at which the sound repeats of its own accord)."""
    n = len(y); out = dict(found=False, n=n, events=[], own=[], left_out=[])
    if n < SR // 4: return out
    pk = float(np.abs(y).max()); quiet = (np.abs(y) <= 16 * TOL).all(1)
    if pk <= 16 * TOL: return out
    S = sorted(stretches(y, table(y), quiet)); dull = dull_ones(y, quiet)
    O = own_way(y, S, quiet); out['own'] = sorted(O); out['stretches'] = S; ev = []; prev_end = -1; prev_L = 0
    for lo, hi, L in S:
        if L in O: continue
        lo2, hi2 = edges(dull, lo, hi, L); s0 = prev_end if lo <= prev_end <= lo2 else lo      # a slice that follows straight on another event starts where that one ends
        if lo < prev_end and hi + L - prev_end >= (1 + KEEP) * L: s0 = prev_end                  # ... also when the sound before it happens to go the same way for a few samples (the ends of two fades; a held note of a machine): what is left of the stretch after the event before is an event of its own
        end = ends_at(s0, L, hi, hi2)
        nxt = []                                                         # another event that starts a whole number of plays on: this one ends there (the sound may go on being equal for a while, where both slices begin alike)
        for q in S:
            if lo < q[0] <= hi + L:
                k = max(int(np.ceil((q[0] - s0) / L)), 1); qa = edges(dull, q[0], q[1], q[2])[0]; w = s0 + k * L
                if q[0] <= w <= qa and (q[2] == L or hi < w): nxt.append(w)      # (one of another slice length: only within the last play)
        if nxt: end = min(nxt)
        whole = s0 + L * int(np.ceil((end - s0) / L - 0.02))            # the last play is taken as whole when what is missing of it is silence (in it and in the slice)
        if end < whole <= n and whole - end <= L // 2 and bool((np.abs(y[end:whole]) <= 16 * TOL).all()) and bool((np.abs(y[end - L:whole - L]) <= 16 * TOL).all()): end = whole
        end = min(end, n)
        if not (end - s0 >= (1 + KEEP) * L or end >= n): continue
        if s0 < prev_end: out['left_out'].append((int(s0), int(L), int(end), 'inside the event before')); continue     # two events cannot lie in each other: the one that starts first is the event, the other an echo of it (see the notes)
        if tone(y, s0, L, end): out['left_out'].append((int(s0), int(L), int(end), 'a steady wave')); continue
        why = 'every play stands by itself in silence' if apart(y, s0, L, end) else goes_on(y, s0, L, end)
        if why: out['left_out'].append((int(s0), int(L), int(end), why)); continue
        ev.append((int(s0), int(L), int(end))); prev_end = end; prev_L = L
    if not ev: return out
    cover = sum(e - s0 - L for s0, L, e in ev) / n; q = [s0 + j * L for s0, L, e in ev for j in range(1, int(np.ceil((e - s0) / L))) if s0 + j * L + 8 < n]
    q = [t for t in q if float(np.abs(y[t:t + FADE]).max()) > 64 * TOL]                                        # the plays that do not begin in silence anyway
    at0 = [bool((np.abs(y[t]) <= 16 * TOL).all()) for t in q]                                                    # a play that starts at nothing and comes up at once: a fade
    out.update(events=ev, share=float(cover), fades=bool(len(q) > 0 and np.mean(at0) >= 0.9), found=True)
    return out

def scaled(events, s, n):
    """The events with Amount at s: the stretch that the extra plays of each slice cover, times s (to whole plays), nothing beyond the sound's end."""
    out = []
    for s0, L, end in events:
        e = end if abs(s - 1.0) < 1e-12 else s0 + L * int(np.floor(1 + ((end - s0) / L - 1) * s + 0.5)); e = min(e, n)
        if e > s0 + L and s0 + L <= n: out.append((s0, L, e))
    return out
def slice_of(z, s0, L, fades):
    sl = z[s0:s0 + L].copy()
    if fades and L >= 2 * FADE: r = (np.arange(FADE) / FADE)[:, None]; sl[:FADE] *= r; sl[L - FADE:] *= r[::-1]
    return sl
def play(z, events, fades):
    """z with every event played: the slice from s0 on, again and again over what follows it, up to the event's end."""
    o = z.copy(); n = len(z); done = 0
    for s0, L, end in events:
        if s0 < done or s0 + L > n: continue                             # an event that starts inside the one before (after a change of the fader): left out
        sl = slice_of(z, s0, L, fades); e = min(end, n); k = e - s0; o[s0:e] = np.tile(sl, (k // L + 1, 1))[:k]; done = e
    return o
def onto(z, f, s=1.0):
    if not f.get('found') or s < 1e-12: return z.copy()
    return play(z, scaled(f['events'], s, len(z)), f['fades'])
def own(y, f, s=1.0):
    """The found sound with Amount at s. Right of the middle: each slice is played more often (over what follows).
    Left of it: fewer times, and since what the repeats covered is gone, the sound closes up and ends early."""
    if not f.get('found') or abs(s - 1.0) < 1e-12: return y.copy()
    n = len(y); ev = scaled(f['events'], s, n)
    if s > 1.0:
        o = y.copy(); was = {(a, b): c for a, b, c in f['events']}
        for s0, L, e in ev:
            e0 = was[(s0, L)]
            if e > e0: k = e - e0; o[e0:e] = np.tile(y[s0:s0 + L], (k // L + 2, 1))[(e0 - s0) % L:(e0 - s0) % L + k]
        return o
    keep = np.ones(n, bool); now = {(a, b): c for a, b, c in ev}
    for s0, L, e0 in f['events']: keep[now.get((s0, L), s0 + L):e0] = False
    o = np.zeros_like(y); k = int(keep.sum()); o[:k] = y[keep]; return o
