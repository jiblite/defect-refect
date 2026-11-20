# The fixed-filter engine, prototype: read the filter's curve off a sound (the reader's nets), take it out, put it on another sound.
import numpy as np, scipy.signal as ss
import ffeat, filt, reader
SR = 48000; FC = ffeat.FC; BOOST = 40.0; CUT = 80.0; NOISE_LIFT = 12.0; MAXBOOST = 24.0
SKIRT = True       # the second look only adds where the first read a deep cut (the reader was taught curves held at -40 dB: a skirt runs on under that)
SILENT_NEAR = 6.0; SILENT_FROM = 12.0      # a band holds nothing when its level is within 6 dB of the lowest the reader is given; the skirt runs on there once the curve is 12 dB down
def skirt_on(L, E):
    """Where a filter has cut everything, the reader has nothing to read and its answer drifts back towards no cut at all. A
    filter's skirt runs on: in a band that holds nothing, the curve keeps the lowest point it reached on the way out from
    its highest point (the pass band), once that is SILENT_FROM dB down."""
    X, ref = ffeat.curves6(np.asarray(E, np.float64)); silent = X[0] <= ffeat.FLOOR + SILENT_NEAR; L = np.array(L, float); top = int(np.argmax(L))
    for bands in (range(top + 1, len(L)), range(top - 1, -1, -1)):
        m = L[top]
        for b in bands:
            m = min(m, L[b])
            if silent[b] and m <= -SILENT_FROM: L[b] = m
    return L
def together(C, how):
    """Several nets' curves (nets, bands) as one. least (what the app does): band by band the one nearest to zero, and zero
    where they do not agree on the sign: what only one of them reads is not taken. mean: their mean."""
    C = np.asarray(C)
    if how == 'mean' or len(C) == 1: return C.mean(0)
    lo = C.min(0); hi = C.max(0); return np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0))
def read(E, nets, rounds=2, how='least', plain=False):
    """E (frames, bands) -> (curve dB per band, sureness, how far the nets lie apart per band in the first look). nets: the
    app's model file, or the trainer's files. rounds 2: the read curve is taken off and the rest read again."""
    m = reader.model(nets); tot = np.zeros(E.shape[1]); e = E.astype(np.float64); first = None; apart = None
    for r in range(rounds):
        X, ref = ffeat.curves6(e); C, T = m.run(X); c = together(C, how)
        if first is None: first = float(T.mean()); apart = C.max(0) - C.min(0)
        if r > 0 and SKIRT: c = c * np.clip((-tot - 24.0) / 12.0, 0.0, 1.0)
        tot += c; e = E.astype(np.float64) * 10 ** (-tot / 10)
    L = np.clip(tot, -CUT, MAXBOOST); return (L if plain else skirt_on(L, E)), first, apart
def dense(L, f):
    """The curve at any frequencies f (Hz): straight lines between the bands on a log-frequency axis; beyond the first and the
    last band it runs on with the slope it has over the half octave there (a filter's skirt goes on), never upwards."""
    lf = np.log2(np.maximum(np.asarray(f, float), 1.0)); lb = np.log2(FC); out = np.interp(lf, lb, L)
    s_lo = (L[6] - L[0]) / (lb[6] - lb[0]); s_hi = (L[-1] - L[-7]) / (lb[-1] - lb[-7]); lo = lf < lb[0]; hi = lf > lb[-1]
    out[lo] = L[0] - max(s_lo, 0.0) * (lb[0] - lf[lo]); out[hi] = L[-1] + min(s_hi, 0.0) * (lf[hi] - lb[-1]); return out
def lift_cap(E, L):
    """How far each band may be lifted when the filter is taken out: BOOST, but NOISE_LIFT in a band that was cut and holds
    still while the bands that were left alone move: such a band is noise that came after the filter."""
    X, ref = ffeat.curves6(E.astype(np.float64)); spread = X[4] - X[2]; heard = X[0] > X[0].max() - 50.0; passb = (np.abs(L) < 3.0) & heard; cap = np.full(len(L), BOOST)
    if not passb.any(): return cap
    moves = float(np.median(spread[passb]))
    if moves >= 14.0: cap[(L < -3.0) & (spread <= 0.5 * moves)] = NOISE_LIFT
    return cap
def softcap(v, cap):
    k = 0.75 * cap; return np.where(v <= k, v, k + (cap - k) * np.tanh((v - k) / np.maximum(cap - k, 1e-9)))
def find(y, nets, rounds=2, how='least', E=None, plain=False):
    """plain: the curve as the nets read it, without the skirt running on where the sound holds nothing (for the studies)."""
    E = ffeat.energies(y) if E is None else E; L, sure, apart = read(E, nets, rounds, how, plain); return dict(curve=L, sure=sure, apart=apart, E=E, lift=lift_cap(E, L))
def moved(L, octaves):
    """A curve moved along the frequency axis (up when positive); what comes in from beyond the ends is the end's own value."""
    if octaves == 0.0: return L
    pos = np.clip(np.arange(len(L)) - 12.0 * octaves, 0.0, len(L) - 1.0); i = np.minimum(pos.astype(int), len(L) - 2); t = pos - i; return L[i] + (L[i + 1] - L[i]) * t
def shaped(L, octaves=0.0, resonance=1.0):
    """The found curve as the Cutoff and Resonance faders shape it."""
    L = moved(np.asarray(L, float), octaves); return np.where(L > 0, L * min(max(resonance, 0.0), 2.0), L)
def own(y, F, s=0.0, octaves=0.0, resonance=1.0, boost=BOOST, capped=True):
    """The sound itself: the found filter out and the one the faders give in (s 0 = out, 1 = untouched, 2 = twice)."""
    v = min(max(s, 0.0), 2.0) * shaped(F['curve'], octaves, resonance) - F['curve']; cap = np.minimum(F['lift'], boost) if capped else np.full(len(v), boost); L = np.where(v > 0, softcap(v, cap), v)
    return filt.apply(y, lambda f: np.clip(dense(L, f), -CUT, BOOST))
def onto(b, F, s=1.0, octaves=0.0, resonance=1.0):
    L = shaped(F['curve'], octaves, resonance) * min(s, 2.0); return filt.apply(b, lambda f: np.clip(dense(L, f), -CUT, MAXBOOST))
