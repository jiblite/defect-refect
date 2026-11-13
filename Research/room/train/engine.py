# The reverb engine as the app will run it (the reference for the C++ port):
#   analyse(x)            -> what was found on a sound: the "room or sound" gains and the room in numbers
#   room_ir(an, ...)      -> a stereo impulse response of the found room (also what Export IR writes)
#   one_sound(x, an, ...) -> the sound with its own reverb turned down / up, and optionally reshaped (length, brightness)
#   onto(b, an, ...)      -> the found room put on a second sound
import sys
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, scipy.signal as ss, common as cm, render, roomnum, roomcap as rc, predict
KMAX = 1500                                   # longest response: 16 s
def gauss(n, seed):
    """n standard normal numbers from a counter-based generator that is easy to repeat exactly anywhere (splitmix64 on a
    counter, pairs turned into normals by Box-Muller)."""
    m = (n + 1) // 2; i = np.arange(1, 2 * m + 1, dtype=np.uint64)
    with np.errstate(over='ignore'):
        z = np.uint64(seed) + i * np.uint64(0x9E3779B97F4A7C15); z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9); z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB); z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0); u1 = u[0::2]; u2 = u[1::2]; r = np.sqrt(-2.0 * np.log(u1))
    return np.stack([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)], 1).reshape(-1)[:n]
_idx = np.clip(np.searchsorted(cm.EDGES, np.arange(cm.NF // 2 + 1), side='right') - 1, 0, cm.NB - 1)       # the band each FFT bin belongs to
def synth_ir(W, unshared=1.0, seed=1):
    """A stereo impulse response tail (n,2) whose band power follows the energy response W (bands, lags) lag by lag: noise
    shaped frame by frame. The two channels share a part so that the unshared share of the tail is `unshared` (0..1, per
    band or one number). The tail starts 10.7 ms after the direct sound (which is not part of it)."""
    L = W.shape[1]; n = L * cm.HOP + cm.NF; u = np.clip(np.broadcast_to(np.asarray(unshared, float), (cm.NB,)), 0.0, 1.0)
    com = gauss(n, seed); ind = np.stack([gauss(n, seed + 1000003), gauss(n, seed + 2000003)], 1); Xc = cm.fwd(np.stack([com, com], 1))[0]; Xi = cm.fwd(ind)
    ub = u[_idx][None, :]; X = np.stack([np.sqrt(1 - ub) * Xc + np.sqrt(ub) * Xi[c] for c in range(2)])                # (2, T, bins): noise with the wanted left/right relation
    click = np.zeros((n, 2)); click[cm.NF] = 1.0; Pk = cm.bandpow(cm.fwd(click)); Pc = Pk.sum(0) / 2.0; c0 = int(np.argmax(Pk.sum(1)))       # a unit click: its band power per channel, and the frame where it sits
    T = X.shape[1]; tgt = np.zeros((T, cm.NB)); Lc = min(L, T - c0); tgt[c0:c0 + Lc] = W[:, :Lc].T
    P = (np.abs(X) ** 2) @ cm.BM.T; G = np.sqrt(tgt[None] * Pc[None, None, :] / np.maximum(P, 1e-30)); Gf = G[:, :, _idx]; Gf[:, :, :cm.EDGES[0]] = Gf[:, :, cm.EDGES[0]:cm.EDGES[0] + 1]; Gf[:, :, cm.EDGES[-1]:] = 0.0
    h = cm.inv(X * Gf, n)
    # second pass: overlapping frames blur the shaping a little, so each band is brought to its exact energy (strength x a unit click's)
    Xh = cm.fwd(h); have = ((np.abs(Xh) ** 2) @ cm.BM.T).sum(1); want = W[:, :Lc].sum(1)[None, :] * Pc[None, :]; cf = np.sqrt(want / np.maximum(have, 1e-30))[:, _idx]; cf[:, cm.EDGES[-1]:] = 0.0
    h = cm.inv(Xh * cf[:, None, :], n); return h[cm.NF - cm.HOP:]
def expo(kdb, T60, drop_db=70.0):
    """Energy response (bands, lags): per band a plain exponential decay with decay time T60 (s) holding 10^(kdb/10) in all;
    long enough for the slowest band to fall by drop_db."""
    Tm = float(np.nanmax(np.where(np.isfinite(T60), T60, 0.05))); L = int(np.clip(np.ceil(Tm * drop_db / 60.0 / cm.DT) + 8, 24, KMAX)); k = np.arange(L); W = np.zeros((cm.NB, L))
    for b in range(cm.NB):
        if not np.isfinite(T60[b]) or not np.isfinite(kdb[b]): continue
        r = 10 ** (-6.0 * cm.DT / max(T60[b], 0.03)); w = r ** k; W[b] = 10 ** (kdb[b] / 10.0) * w / w.sum()
    return W
def analyse(x, model=None, g=None):
    """x (n,2) at 48 kHz. Returns dict: LE (band levels, dB), g (gains, dB), found (bool), and when a room was found:
    kdb, T60 (per band), ev (evidence per band), width (unshared share of the tail per band, 0 = mono .. 1 = fully wide)."""
    ft = cm.features(x); g = predict.gains(ft, model) if g is None else g; an = dict(LE=ft['LE'], g=g, found=False); rn = roomnum.room_numbers(ft['LE'], g)       # (g given: a mask from elsewhere, for measurements)
    if rn is None: return an
    kdb, T60, ev = rn; U = 10 ** (np.clip(ft['U256'], -40, 0) / 10); tail = (g < -12.0) & (ft['LE'] > ft['LE'].max() - 50.0)
    u = np.array([np.median(U[tail[:, b], b]) if tail[:, b].sum() >= 10 else np.nan for b in range(cm.NB)]); u = np.where(np.isfinite(u), u, np.nanmedian(u) if np.isfinite(u).any() else 1.0)
    an.update(found=True, kdb=kdb, T60=T60, ev=ev, width=np.clip(u * 1.15, 0.0, 1.0)); return an
def shaped(an, length=1.0, bright=1.0, width=1.0):
    """The found numbers with the faders applied (each 0..2, 1 = as found): Length scales the decay time (the level right
    after the direct sound stays where it was); Brightness tilts the decay time and the strength of the upper bands against
    the lower ones (pivot 1 kHz); Width scales the tail's width."""
    oct_ = np.log2(cm.FC41 / 1000.0); tilt = (bright - 1.0)                                       # -1 .. +1
    T60 = np.clip(an['T60'] * max(length, 0.02) * 2.0 ** (0.35 * tilt * oct_), 0.03, 20.0); r0 = 10 ** (-6.0 * cm.DT / an['T60']); r1 = 10 ** (-6.0 * cm.DT / T60)
    kdb = an['kdb'] + 3.0 * tilt * oct_ + 10 * np.log10((1.0 - r0) / (1.0 - r1))       # the room keeps its level right after the direct sound: a longer room holds more energy in all, a shorter one less
    return kdb, T60, np.clip(an['width'] * width, 0.0, 1.0)
def room_ir(an, length=1.0, bright=1.0, width=1.0, seed=1):
    kdb, T60, u = shaped(an, length, bright, width); return synth_ir(expo(kdb, T60), u, seed=seed)
def lean_db(an):
    """A room rebuilt from numbers is one even wash; a real short room is mostly a few early reflections, which the ear (and
    the scoreboard) takes as part of the sound itself. At the same energy the copy therefore comes out wetter than the real
    thing when the room is short: on a second sound it goes on 4 dB quieter for mid-band decay times up to 1 s, less and
    less up to 2 s, and as found from there on. (Measured: +5 points on short and middle rooms, long rooms want none.)"""
    Tm = float(np.median(an['T60'][roomnum.MID])); return -4.0 * float(np.clip(1.0 - np.log2(max(Tm, 1e-3)), 0.0, 1.0))
def onto(b, an, amount=1.0, length=1.0, bright=1.0, width=1.0):
    """The found room on a second sound: amount 0 = none, 1 = as found, 2 = twice as loud."""
    if not an['found'] or amount <= 0: return b.copy()
    return rc.apply_room(b, room_ir(an, length, bright, width), amount * 10 ** (lean_db(an) / 20.0))
FMAX = 16.0                                   # the room part of a cell is never made more than 12 dB louder
def reshape_factor(an, T1, dk_db=0.0):
    """How much stronger (power) the room part of every band and moment has to be for the room to decay with T1 (s, per
    band) instead of the found decay time, starting from the same level: the direct sound as the mask has it, run through
    both decays, one against the other. dk_db: a level change of the room part per band on top."""
    P0 = roomnum.smear(an['LE'], an['g'], an['T60']); P1 = roomnum.smear(an['LE'], an['g'], T1); r0 = 10 ** (-6.0 * cm.DT / an['T60']); r1 = 10 ** (-6.0 * cm.DT / T1)
    F = np.where(P0 > 1e-9 * P0.max(), P1 / np.maximum(P0, 1e-300), 1.0) * ((1.0 - r0) / (1.0 - r1) * 10 ** (np.broadcast_to(np.asarray(dk_db, float), (cm.NB,)) / 10.0))[None, :]
    return np.clip(F, 0.0, FMAX)
def one_sound(x, an, amount=1.0, length=1.0, bright=1.0, width=1.0):
    """The sound itself: amount 0 = reverb out, 1 = untouched, 2 = reverb twice as loud. Length, Brightness and Width
    reshape the sound's OWN reverb (nothing synthetic is added): the room part of every band and moment is turned up or
    down so that it decays with the new decay time from the same starting level, tilted and widened as asked."""
    if not an['found'] or (abs(length - 1.0) < 1e-9 and abs(bright - 1.0) < 1e-9 and abs(width - 1.0) < 1e-9): return render.render(x, an['g'], amount)
    oct_ = np.log2(cm.FC41 / 1000.0); tilt = bright - 1.0; T1 = np.clip(an['T60'] * max(length, 0.02) * 2.0 ** (0.35 * tilt * oct_), 0.03, 20.0)
    return render.reshaped(x, an['g'], amount, reshape_factor(an, T1, 3.0 * tilt * oct_), float(np.clip(width, 0.0, 2.0)))
HALF, DEEP = 0.5, 1.5      # the app's Amount fader on the sound itself (see fader)
def deeper(g, b):
    """The model's answer made bolder (b > 0) or gentler (b < 0): the same answer with its "how sure" shifted, so cells it
    is sure about stay where they are and the doubtful ones move most."""
    p = np.clip(-g / 30.0, 1e-6, 1.0 - 1e-6); return -30.0 / (1.0 + np.exp(-(np.log(p / (1.0 - p)) + b)))
def fader(x, an, s, length=1.0, bright=1.0, width=1.0):
    """The Amount fader as the app has it on the sound itself, s 0..2: middle (1) = untouched; from the middle to HALF way
    left the model's own answer comes in (at HALF it is in full); from there to far left it is made bolder, up to depth
    DEEP at 0 (how deep is right depends on the sound: the ear picks); right of the middle = more reverb (one_sound)."""
    if s >= 1.0: return one_sound(x, an, s, length, bright, width)
    if s >= HALF: return one_sound(x, an, (s - HALF) / (1.0 - HALF), length, bright, width)
    a2 = dict(an); a2['g'] = deeper(an['g'], DEEP * (HALF - s) / HALF); return one_sound(x, a2, 0.0, length, bright, width)
def one_sound_rebuilt(x, an, amount=1.0, length=1.0, bright=1.0, width=1.0):
    """(The earlier way, kept for comparison: the sound's own reverb gives way to the found room rebuilt with the new settings.)"""
    moved = abs(np.log2(max(length, 0.02))) / 0.5 + abs(bright - 1.0) / 0.5 + abs(width - 1.0) / 0.5; w = float(np.clip(moved, 0.0, 1.0)) if an['found'] else 0.0
    if w <= 0: return render.render(x, an['g'], amount)       # (adding the found room on top for "more" was tried: far worse than making the room parts louder)
    d = render.render(x, an['g'], 0.0); new = rc.apply_room(d, room_ir(an, length, bright, width), 1.0) - d                         # the rebuilt room alone
    own = render.render(x, an['g'], amount * (1.0 - w)) if amount * (1.0 - w) <= 1.0 else render.render(x, an['g'], 1.0)
    return own + amount * w * new
