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
EXTRA = int(0.030 * SR)          # an event's repeats have to cover this much at least (two cycles of a low note that happen to be equal are not an event)
LOOP, LOOP_SHARE = int(0.3 * SR), 0.3            # a slice longer than this that 30 % of the whole sound repeats is the sound's own way (a loop), not an event
TOL = 2.0 ** -18                 # two samples are "the same" if they are this close (a 24-bit file with dither still passes)
FADE = 96                        # samples of the fade at each end of a slice when the found slices start and end at nothing (2 ms)
KEEP = 0.9                       # a stretch counts if it is at least this share of its slice long

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

def stretches(y, prev):
    """The stretches where the sound is what it was L samples before: list of (lo, hi, L) in terms of the earlier
    sample: samples lo + L .. hi + L - 1 equal samples lo .. hi - 1. Silence equals silence, which proves nothing: a
    stretch is cut where a whole slice of it is silent, and it starts and ends no further out than one slice from its
    first and last live sample."""
    n = len(y); t = np.where(prev > 0)[0]; out = []
    if not len(t): return out
    lag = t - (prev[t] - 1); ok = (lag >= LMIN) & (lag <= LMAX); t, lag = t[ok], lag[ok]
    if not len(t): return out
    cut = np.where((np.diff(t) > WIN) | (np.diff(lag) != 0))[0] + 1; starts = np.concatenate([[0], cut]); ends = np.concatenate([cut, [len(t)]]); quiet = (np.abs(y) <= 16 * TOL).all(1); seen = {}
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
            live = int((~quiet[int(fl) + L:int(ll) + L + 1]).sum())
            if (b0 - a0 >= KEEP * L or (b0 + L == n and b0 - a0 >= 0.25 * L)) and b0 - a0 >= EXTRA and live >= max(0.2 * L, 4 * WIN): out.append((int(a0), int(b0), L))     # cut off by the end of the sound: a quarter of a slice is enough
    lv = ~quiet; nl = max(int(lv.sum()), 1); loops = set()
    for L in set(q[2] for q in out):
        if L > LOOP and float(((np.abs(y[L:] - y[:-L]) <= TOL).all(1) & lv[L:]).sum()) / nl >= LOOP_SHARE: loops.add(L)
    return [q for q in out if q[2] not in loops]

def edges(y, lo, hi, L):
    """How far the start of a stretch may lie after lo, and its end before hi: the samples there that are next to
    nothing on both sides (they are equal whatever was played, and say nothing)."""
    n = len(y); q = 16 * TOL; a = lo
    while a < lo + 16 and a + L < n and float(np.abs(y[a]).max()) <= q and float(np.abs(y[a + L]).max()) <= q: a += 1
    b = hi
    while b > hi - 16 and b - 1 > a and float(np.abs(y[b - 1]).max()) <= q and float(np.abs(y[b - 1 + L]).max()) <= q: b -= 1
    return a, b
def find(y):
    """dict(found, events: list of (s0, L, end): the slice s0 .. s0 + L is played again and again up to sample end;
    fades: whether the slices start and end at nothing; share: how much of the sound the repeats cover)."""
    n = len(y); out = dict(found=False, n=n, events=[])
    if n < SR // 4: return out
    S = sorted(stretches(y, table(y))); ev = []; prev_end = -1
    for lo, hi, L in S:
        lo2, hi2 = edges(y, lo, hi, L); s0 = prev_end if lo <= prev_end <= lo2 else lo         # a slice that follows straight on another event starts where that one ends
        m = int(np.floor((hi + L - s0) / L + 0.5)); end = s0 + m * L if hi2 + L <= s0 + m * L <= hi + L else hi + L       # whole plays where the samples allow it
        whole = s0 + L * int(np.ceil((end - s0) / L - 0.02))            # the last play is taken as whole when what is missing of it is silence (in it and in the slice)
        if end < whole <= n and whole - end <= L // 2 and bool((np.abs(y[end:whole]) <= 16 * TOL).all()) and bool((np.abs(y[end - L:whole - L]) <= 16 * TOL).all()): end = whole
        end = min(end, n)
        if end - s0 >= 2 * L - L // 10 or end >= n: ev.append((int(s0), int(L), int(end))); prev_end = end
    out['stretches'] = S
    if not ev: return out
    cover = sum(e - s0 - L for s0, L, e in ev) / n; pk = float(np.abs(y).max()) + 1e-30
    edge = [max(float(np.abs(y[s0 + j * L]).max()), float(np.abs(y[min(s0 + (j + 1) * L - 1, n - 1)]).max())) / pk for s0, L, e in ev for j in range(int((e - s0) // L))]
    out.update(events=ev, share=float(cover), fades=bool(np.median(edge) < 0.02), found=True)
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
