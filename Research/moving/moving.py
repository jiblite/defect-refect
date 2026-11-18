# The moving-tone engine's prototype (PHASER, FILTER SWEEP): a filter whose curve moves over time.
#   find(model, x)        what is on a stereo sound: the tone curve per moment and band for each side, where along the sound
#                         it is, which of the two kinds it is and how sure
#   apply(x, F, s, ...)   s = -1: the found curve taken off the sound it was found on; s = +1: put onto another sound at the
#                         same times; anything between and up to +2: scaled (in dB)
# The curve comes from the model (net.py) that reads the band picture (pic.py). A sound longer than 6 s is read in blocks of
# 6 s, every 3 s, each block on its own (as the model was fitted), and the answers are faded into one another.
# How sure: what the model says is not enough by itself (it says "moving filter" with 0.65 to 0.9 on one clean sound in
# seven). A filter sweep counts when its curve is also big; a phaser, whose curve is as small as a sound's own movement,
# only when the model is very sure. And the model reads a curve too small, so where it is sure the curve is taken off, what
# is left is read again and added (rule.py, rule2.py, feat.py, iter.py).
import numpy as np, torch
import pic, net
BLOCK = 282; BHOP = 141                    # frames: 6 s, 3 s
ON = 0.5; GAP = 24; SHORT = 47; RAMP = 5    # a frame counts from this probability; holes under 0.5 s are closed, stretches under 1 s dropped; 0.1 s fades at the ends
SMOOTH = 15                                # the per-frame say-so is averaged over 0.3 s first
PERIODIC = 0.8                             # the curve repeats if it is this like itself one period on
CUT = 0.65                                 # the app's default Sureness: what `sure` is scaled to
PHASER = 0.93                              # the say-so a phaser needs to be shown at that default
SWEEP = 0.81; BIG = 7.4                    # the say-so and the curve's size (dB, inside the stretch) a filter sweep needs
FULL = 94                                  # frames (2 s): a shorter stretch is trusted less, in proportion
KEEP = 0.5                                 # a stretch counts if it is at least this sure against the surest
AGAIN = 0.9; LEFT = 0.87                   # where the say-so was this high the curve is taken off and what is left is read and added: a filter sweep once; a phaser up to
                                           # twice, each time only if the model still says 'moving filter' this clearly of what is left

def model_of(path):
    st = torch.load(path, map_location='cpu', weights_only=False); m = net.UNet(cin=st['cfg']['cin'], ch=tuple(st['cfg']['ch']), bpool=st['cfg']['bpool']); m.load_state_dict(st['state']); m.eval(); return m

def blocks(T):
    if T <= BLOCK: return [(0, T)]
    return [(a, a + BLOCK) for a in list(range(0, T - BLOCK, BHOP)) + [T - BLOCK]]

def read(model, x):
    """One channel -> (curve (T, NB) in dB, probabilities (T, 3): nothing, filter sweep, phaser)."""
    LE = pic.picture(x); T = len(LE); G = np.zeros((T, pic.NB)); P = np.zeros((T, 3)); W = np.zeros(T)
    for a, b in blocks(T):
        with torch.no_grad(): g, c = model(torch.from_numpy(pic.inputs(LE[a:b]))[None])
        n = b - a; w = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)).astype(float)
        G[a:b] += w[:, None] * g[0].numpy().astype(np.float64) * net.GS; P[a:b] += w[:, None] * net.probs(c[0].numpy().astype(np.float64)); W[a:b] += w
    return G / W[:, None], P / W[:, None]

def stretches(on):
    """Frames with a moving filter -> [(first, end)]: holes shorter than GAP closed, stretches shorter than SHORT dropped."""
    T = len(on); sp = []; t = 0
    while t < T:
        if on[t]:
            a = t
            while t < T and on[t]: t += 1
            sp.append([a, t])
        else: t += 1
    j = []
    for a, b in sp:
        if j and a - j[-1][1] < GAP: j[-1][1] = b
        else: j.append([a, b])
    return [(a, b) for a, b in j if b - a >= min(SHORT, T)]

def period(G):
    """The curve's period in frames if it repeats (at least twice within the stretch), else 0."""
    T = len(G); e = (G ** 2).sum(1); best = 0.0; r = np.zeros(T // 2 + 1)
    if T < 48 or e.sum() <= 0: return 0.0
    for L in range(12, T // 2 + 1): r[L] = (G[:-L] * G[L:]).sum() / max(np.sqrt(e[:-L].sum() * e[L:].sum()), 1e-12)
    for L in range(13, T // 2):
        if r[L] >= PERIODIC and r[L] >= r[L - 1] and r[L] >= r[L + 1]:                  # the first lag at which it is like itself again
            d = r[L - 1] - 2 * r[L] + r[L + 1]; return L + (0.5 * (r[L - 1] - r[L + 1]) / d if d < 0 else 0.0)
    return 0.0

def gate(a, n, T):
    """Weights for the n frames from frame a on of a stretch: a 0.1 s fade at each end
    that lies inside a sound of T frames (none at the sound's own start and end)."""
    i = np.arange(n); m = np.ones(n)
    if a > 0: m = np.minimum(m, (i + 1) / RAMP)
    if a + n < T: m = np.minimum(m, (n - i) / RAMP)
    return np.clip(m, 0.0, 1.0)

def scale(v, zero, at, full):
    """0 at `zero`, CUT at `at`, 1 at `full`, straight lines between."""
    return CUT + (1.0 - CUT) * min(1.0, (v - at) / (full - at)) if v >= at else CUT * max(0.0, (v - zero) / (at - zero))

def sureness(kind, say, size, frames):
    """How sure a stretch is, on the app's scale (shown at the default when over CUT)."""
    s = scale(say, ON, PHASER, 1.0) if kind == 'phaser' else min(scale(say, ON, SWEEP, 1.0), scale(size, 0.0, BIG, 2.0 * BIG))
    return s * min(1.0, frames / FULL)

def reading(model, x):
    """Both sides -> (curves (2, T, NB), probabilities (2, T, 3))."""
    same = np.array_equal(x[:, 0], x[:, 1]); r = [read(model, x[:, 0])]; r.append(r[0] if same else read(model, x[:, 1]))
    return np.stack([g for g, _ in r]), np.stack([p for _, p in r])

def find(model, x):
    """x (n, 2). Returns dict: found, kind ('filtersweep' or 'phaser'), sure (0 .. 1, on the app's scale), say (the model's
    say-so inside the stretches), size (the curve's size there, dB), spans [(first frame, end frame)], raw (2, T, NB) the
    curve per side, gate (T,) 1 inside the stretches, G = raw * gate, on (T,) the say-so per frame, period (frames of the
    longest stretch, 0 = the curve does not repeat), reads (how often the curve was read), all (every stretch the model
    named: first, end, kind, say, size, sure)."""
    raw, P = reading(model, x); T = raw.shape[1]
    on = np.maximum(1.0 - P[0][:, 0], 1.0 - P[1][:, 0]); k = min(SMOOTH, T); sm = np.convolve(np.pad(on, (k // 2, k - 1 - k // 2), mode='edge'), np.ones(k) / k, mode='valid')
    sp = stretches(sm >= ON)
    F = dict(found=False, spans=[], on=on, n=len(x), T=T, kind='', sure=0.0, say=0.0, size=0.0, period=0.0, raw=raw, gate=np.zeros(T), G=raw * 0.0, reads=1, all=[])
    if not sp: return F
    for a, b in sp:                                                       # every stretch for itself: which kind, how sure
        kind = 'filtersweep' if P[:, a:b, 1].sum() >= P[:, a:b, 2].sum() else 'phaser'; say = float(on[a:b].mean()); size = float(np.sqrt((raw[:, a:b] ** 2).mean()))
        F['all'].append((a, b, kind, say, size, sureness(kind, say, size, b - a)))
    top = max(F['all'], key=lambda q: q[5]); keep = [q for q in F['all'] if q[2] == top[2] and q[5] >= KEEP * top[5]]       # the surest stretch says which kind; much less sure ones are left alone
    m = np.zeros(T)
    for a, b, *_ in keep: m[a:b] = gate(a, b - a, T)
    inside = m > 0; F.update(found=True, kind=top[2], sure=float(top[5]), spans=[(q[0], q[1]) for q in keep], gate=m, G=raw * m[None, :, None], say=float(on[inside].mean()), size=float(np.sqrt((raw[:, inside] ** 2).mean())))
    while F['say'] >= AGAIN and F['reads'] < (3 if F['kind'] == 'phaser' else 2):   # the model reads the curve too small: take it off, read what is left, add it
        r2, P2 = reading(model, apply(x, F, -1.0)); T2 = min(T, r2.shape[1]); left = float(np.maximum(1.0 - P2[0][:T2, 0], 1.0 - P2[1][:T2, 0])[inside[:T2]].mean())
        if F['kind'] == 'phaser' and left < LEFT: break                          # a phaser's rest is added only while the model is sure there is one
        F['raw'] = F['raw'].copy(); F['raw'][:, :T2] += r2[:, :T2]; F['G'] = F['raw'] * m[None, :, None]; F['reads'] += 1
    a, b = max(F['spans'], key=lambda q: q[1] - q[0]); F['period'] = float(period(F['raw'][0][a:b] + F['raw'][1][a:b]))
    return F

def curve(F, c, T2, speed=1.0):
    """Side c's curve for a sound of T2 frames, moving `speed` times as fast. Every stretch keeps its place in time; inside
    it the curve is read faster or slower from the stretch's start, and when it runs out it carries on: a curve that repeats
    goes round its last period, any other goes back and forth. A stretch that reaches the end of the sound it was found on
    runs on to the end of a longer sound in the same way."""
    raw = F['raw'][c]; T = len(raw)
    if T2 <= T and speed == 1.0: return F['G'][c][:T2]
    out = np.zeros((T2, pic.NB)); P = F['period']
    for k, (a, b) in enumerate(F['spans']):
        e = max(b, T2) if (k == len(F['spans']) - 1 and b >= T - 15) else min(b, T2)
        if e <= a: continue
        t = np.arange(a, e); u = a + (t - a) * speed; L = max(b - 1 - a, 1)
        if 0 < P < L: over = u > b - 1; u[over] = b - 1 - P + np.mod(u[over] - (b - 1), P)
        else: v = np.mod(u - a, 2 * L); u = a + np.where(v <= L, v, 2 * L - v)
        i0 = np.clip(u.astype(int), 0, max(T - 2, 0)); f = np.clip(u - i0, 0.0, 1.0)[:, None]
        out[a:e] = (raw[i0] * (1 - f) + raw[np.minimum(i0 + 1, T - 1)] * f) * gate(a, e - a, T2)[:, None]
    return out

def apply(x, F, s, speed=1.0, own=True, lift=24.0):
    """x (n, 2) with the found curve scaled by s. own: x is the sound the curve was found on (s = -1 takes it off; with another
    speed the found curve goes out and the new one, 1 + s times, comes in). Otherwise s times the curve goes onto x."""
    T2 = len(x) // pic.HOP + 1; out = np.zeros_like(x)
    for c in range(2):
        if own and speed != 1.0: g = (1.0 + s) * curve(F, c, T2, speed) - curve(F, c, T2)
        else: g = s * curve(F, c, T2, speed)
        out[:, c] = pic.render(x[:, c], g, 1.0, lift)
    return out
