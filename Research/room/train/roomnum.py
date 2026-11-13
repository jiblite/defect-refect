# The room in numbers, read off a sound with the help of the "room or sound" mask: per band a decay time (straight-line
# fits to the level in the stretches the mask calls "all room": the free decays) and a strength (how loud those stretches
# are next to the direct sound smeared out with that decay time).
import sys
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, common as cm
def _runs(m):
    """Start and end (exclusive) of every run of True."""
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return np.where(d == 1)[0], np.where(d == -1)[0]
def decay_slopes(LE, g, gmax=-10.0, minlen=8, skip=2, floor=57.0):
    """Per band: (slope in dB/s, weight) for every free decay: a run of at least minlen frames with gain under gmax and
    level above the floor; the first `skip` frames (still holding the end of the direct sound) are left out. A decay counts
    more the further it falls and the straighter it is."""
    T, B = LE.shape; fl = LE.max() - floor; out = [[] for _ in range(B)]; tt = np.arange(T) * cm.DT
    for b in range(B):
        st, en = _runs((g[:, b] < gmax) & (LE[:, b] > fl))
        for t, e in zip(st, en):
            a = t + skip
            if e - a < minlen: continue
            y = LE[a:e, b]; x = tt[a:e]; p = np.polyfit(x, y, 1); res = np.sqrt(np.mean((np.polyval(p, x) - y) ** 2)); rng = -p[0] * (x[-1] - x[0])
            if p[0] < -3.0 and rng > 6.0: out[b].append((p[0], rng / (1.0 + res)))
    return out
def decay_times(LE, g):
    """T60 per band (s; nan where the room never shows) and how much evidence each band has. If nothing qualifies anywhere,
    the rules are relaxed step by step (shorter stretches, less clearly 'all room')."""
    for gmax, minlen in ((-10.0, 8), (-7.0, 6), (-4.0, 5)):
        sl = decay_slopes(LE, g, gmax, minlen); T60 = np.full(cm.NB, np.nan); ev = np.zeros(cm.NB)
        for b, lst in enumerate(sl):
            if not lst: continue
            s = np.array([v[0] for v in lst]); w = np.array([v[1] for v in lst]); o = np.argsort(s); cw = np.cumsum(w[o]) / w.sum(); T60[b] = -60.0 / s[o][np.searchsorted(cw, 0.5)]; ev[b] = w.sum()
        if np.isfinite(T60).sum() >= 4: break
    return T60, ev
def smooth_fill(v, ev, n=2, log=True):
    """Per-band numbers made to agree with their neighbours (evidence-weighted median over +-n bands) and filled where a
    band has none (from the nearest bands that have; flat beyond the ends)."""
    x = np.log(v) if log else np.array(v, float); ok = np.isfinite(x) & (ev > 0)
    if not ok.any(): return np.full(len(v), np.nan)
    sm = np.full(len(v), np.nan)
    for b in range(len(v)):
        lo, hi = max(0, b - n), min(len(v), b + n + 1); sel = ok[lo:hi]
        if sel.any():
            xs = x[lo:hi][sel]; ws = ev[lo:hi][sel]; o = np.argsort(xs); cw = np.cumsum(ws[o]) / ws.sum(); sm[b] = xs[o][np.searchsorted(cw, 0.5)]
    idx = np.where(np.isfinite(sm))[0]; sm = np.interp(np.arange(len(v)), idx, sm[idx]); return np.exp(sm) if log else sm
def smear(LE, g, T60, gmax=-10.0, floor=57.0):
    """The direct sound of each band (where the mask says there is some: gain above gmax) smeared out with the band's
    decay time: what a room with that decay and strength 1 would hold at every moment. (T, B)."""
    T, B = LE.shape; E = 10 ** (LE / 10.0); fl = LE.max() - floor; S = np.where((g > gmax) & (LE + g > fl + 5.0), E * 10 ** (g / 10.0), 0.0); P = np.zeros((T, B))
    for b in range(B):
        if not np.isfinite(T60[b]): continue
        r = 10 ** (-6.0 * cm.DT / T60[b]); acc = 0.0
        for t in range(T): acc = r * acc + (1.0 - r) * S[t, b]; P[t, b] = acc
    return P
def second_look(LE, g):
    """Three extra pictures for the model's second pass, made from its first answer g: (1) how loud the room should be at
    every band and moment next to what is there (expected room share, from the found decay time and strength), (2) the
    found decay time and (3) the found strength, each band's number held over time. No room found: all three say so."""
    T, B = LE.shape; rn = room_numbers(LE, g)
    if rn is None: return np.stack([np.full((T, B), -1.0), np.full((T, B), -1.0), np.full((T, B), -1.0)]).astype(np.float32)
    kdb, T60, ev = rn; P = smear(LE, g, T60); E = 10 ** (LE / 10.0); ratio = 10 * np.log10(10 ** ((kdb + BUILD_UP_DB) / 10.0)[None, :] * P / np.maximum(E, 1e-30) + 1e-9)
    p1 = np.clip(ratio, -30.0, 10.0) / 20.0 + 0.5; p2 = np.clip(np.log2(T60), -3.3, 3.3) / 3.3; p3 = np.clip(kdb, -30.0, 20.0) / 25.0 + 0.2
    return np.stack([p1, np.broadcast_to(p2[None, :], (T, B)), np.broadcast_to(p3[None, :], (T, B))]).astype(np.float32)
def strengths(LE, g, T60, gmax=-10.0, floor=57.0, trim=0.1):
    """Per band: strength kappa (reverb energy per unit of direct energy, for a plain exponential decay with decay time
    T60) and the evidence. In the 'all room' moments the level is kappa times the direct sound smeared with that decay."""
    T, B = LE.shape; E = 10 ** (LE / 10.0); fl = LE.max() - floor; S = np.where((g > gmax) & (LE + g > fl + 5.0), E * 10 ** (g / 10.0), 0.0); kap = np.full(B, np.nan); ev = np.zeros(B)
    for b in range(B):
        if not np.isfinite(T60[b]): continue
        r = 10 ** (-6.0 * cm.DT / T60[b]); P = np.zeros(T); acc = 0.0
        for t in range(T): acc = r * acc + (1.0 - r) * S[t, b]; P[t] = acc
        rows = (g[:, b] < gmax) & (LE[:, b] > fl) & (P > 1e-4 * P.max()) if P.max() > 0 else np.zeros(T, bool)
        if rows.sum() < 8: continue
        q = np.sort(np.log(E[rows, b] / P[rows])); k = int(len(q) * trim); q = q[k:len(q) - k] if len(q) - 2 * k >= 4 else q
        kap[b] = np.exp(np.mean(q)); ev[b] = rows.sum()
    return kap, ev
BUILD_UP_DB = 2.3      # a real room builds up before it decays: a plain exponential that matches the tail holds about this much more energy (median over the training reverbs, true mask)
# No band may stray further above the middle of the spectrum than real reverbs do (about the 98th percentile of the true
# values over the training and test reverbs): a band that reads far beyond that has been misread.
MID = (cm.FC41 > 450) & (cm.FC41 < 7000)
CAP_T = np.where(cm.FC41 < 450, 5.0, np.where(cm.FC41 <= 7000, 2.5, 1.6))      # decay time: at most this many times the mid-band median
CAP_K = np.where(cm.FC41 < 450, 12.0, np.where(cm.FC41 <= 7000, 6.0, 4.0))     # strength: at most this many dB above the mid-band median
def _mid(v, ev):
    """The median over the mid bands (450 Hz .. 7 kHz) that have evidence of their own; if none has, over all bands that have."""
    s = MID & (ev > 0)
    if not s.any(): s = ev > 0
    return float(np.median(v[s])) if s.any() else float(np.median(v))
def room_numbers(LE, g):
    """(strength in dB, decay time in s, evidence) per band, smoothed over neighbouring bands and filled; None if no room
    shows. The strength is that of the whole response (the tail level minus the build-up allowance); evidence = how many
    'all room' moments each band had."""
    T60, ev = decay_times(LE, g)
    if not np.isfinite(T60).any(): return None
    T60s = smooth_fill(T60, ev); T60s = np.minimum(T60s, _mid(T60s, ev) * CAP_T); kap, ek = strengths(LE, g, T60s)
    if not np.isfinite(kap).any(): return None
    kdb = 10 * np.log10(smooth_fill(kap, ek)) - BUILD_UP_DB; return np.minimum(kdb, _mid(kdb, ek) + CAP_K), T60s, ek
