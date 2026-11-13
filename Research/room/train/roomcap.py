# Room capture, prototype: from the separated band levels (direct part, reverb part) to the room's energy response per band
# (how much reverb power follows a unit of direct power, lag by lag), its numbers (decay time, strength, pre-delay), and a
# stereo impulse response with that response, to put the same room on another sound.
import sys
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, scipy.signal as ss, scipy.optimize as so, common as cm
K = 420                                                             # lags followed: 4.5 s
KNOTS = np.array([0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 34, 44, 57, 74, 96, 125, 162, 210, 273, 355, 419])
EARLY = 6                                                           # the first lags (64 ms) are free: pre-delay, build-up, single early reflections
def _basis():
    """What a response is built from: one free value for each of the first EARLY lags, then a part that can only fall:
    values at the later knots that never rise, straight lines between them (as a mix of 'full until this knot, then down to
    nothing at the next' shapes with non-negative weights)."""
    kn = KNOTS[KNOTS >= EARLY]; P = []; k = np.arange(K)
    for e in range(EARLY): v = np.zeros(K); v[e] = 1.0; P.append(v)
    for i, m in enumerate(kn):
        nx = kn[i + 1] if i + 1 < len(kn) else m + 1; v = np.where(k < EARLY, 0.0, np.where(k <= m, 1.0, np.clip((nx - k) / max(nx - m, 1), 0.0, 1.0))); P.append(v)
    return np.array(P)
PHI = _basis()
def _nnls(A, y):
    """Non-negative least squares that always comes back: columns brought to the same size first, and a slower, safe
    solver when the quick one gives up."""
    nrm = np.sqrt((A ** 2).sum(0)); nrm = np.where(nrm > 0, nrm, 1.0); An = A / nrm
    try: c = so.nnls(An, y, maxiter=30 * A.shape[1])[0]
    except RuntimeError: c = so.lsq_linear(An, y, bounds=(0.0, np.inf), max_iter=200).x
    return np.maximum(c, 0.0) / nrm
def tracks(LE, g_db, floor=60.0):
    """Band power of the direct part and of the reverb part from the level picture and the gains (both dB), the cells
    that can be used at all, and the bands whose direct sound is strong enough to show its room.
    The gains have two stops that are not answers: they never go under -30 dB, and a direct part under the level floor
    (60 dB under the loudest cell) is given as the floor itself. Both would read as a faint direct sound that is always
    there, so such cells count as 'no direct sound'; a band where much of the direct sound sits that low cannot be read."""
    E = 10 ** (LE / 10.0); S0 = E * 10 ** (g_db / 10.0); R = np.maximum(E - S0, 0.0); fl = LE.max() - floor
    S = np.where((LE + g_db <= fl + 2.0) | (g_db <= -29.5), 0.0, S0); strong = np.where(LE + g_db >= fl + 15.0, S, 0.0)
    return S, R, LE > fl + 3.0, strong.sum(0) >= 0.9 * np.maximum(S0.sum(0), 1e-30)
def energy_response(S, R, valid=None, band_ok=None, mode='pred', iters=3, c1=1.0, tie=0, soft=4.0, floor_db=-40.0, clean=True, fill=True):
    """W (bands, K): reverb power at lag k per unit of direct power, per band, by non-negative least squares.
    The reverb can only be read where it is not under the direct sound, and its power wobbles around any smooth curve, so:
    the moments used are picked by what the fit itself EXPECTS there (expected reverb at least c1 times the direct part),
    never by what was seen (that would pick the upward wobbles), each moment counts relative to its expected level (down to
    floor_db under the band's loudest), and the first lags, which sit inside the direct sound's own frames and cannot be
    seen at all, take the value of lag `tie`. mode 'inst' is the old way (picked by what was seen)."""
    T, B = S.shape; W = np.zeros((B, K)); top = max(S.max(), R.max()); E = S + R; trust = np.zeros(B, bool)
    P = PHI.copy()
    if tie > 0: P = np.vstack([PHI[:tie + 1].sum(0)[None], PHI[tie + 1:]])
    for b in range(B):
        s = S[:, b]; r = R[:, b]; e = E[:, b]
        if r.sum() <= 1e-9 * top or s.sum() <= 1e-9 * top: continue
        A = np.stack([np.convolve(s, P[j])[:T] for j in range(len(P))], 1); live = e > 1e-6 * e.max()
        if valid is not None: live = live & valid[:, b]
        rs = np.convolve(r, np.ones(5) / 5, 'same')
        if mode == 'inst':
            seen = (r >= 0.6 * e) & live
            if seen.sum() < 12: continue
            wt = (1.0 / np.sqrt(rs + 0.001 * r.max() + 1e-30)) * seen; c = _nnls(A * wt[:, None], r * wt)
        else:
            if live.sum() < 12: continue
            wt = (1.0 / np.sqrt(rs + 0.001 * r.max() + 1e-30)) * live; c = _nnls(A * wt[:, None], r * wt); fl = e.max() * 10 ** (floor_db / 10.0)   # start: every moment
            for it in range(iters):
                p = A @ c; cc = c1; rows = (p >= cc * s) & live
                while rows.sum() < 30 and cc > 1e-3: cc *= 0.25; rows = (p >= cc * s) & live                              # a room that never shows by itself: take the moments where it shows most
                if rows.sum() < 30: rows = live
                trust[b] = cc == c1 and (band_ok is None or band_ok[b])
                wt = rows / (p + fl); Aw = A * wt[:, None]; yw = r * wt
                if soft > 0 and tie == 0:
                    # the first two lags sit inside the direct sound's own frames: unless the moments used can tell (a room louder than the sound), lag 1 follows lag 2 and lag 0 is half of lag 1 (what rooms and reverbs do on average)
                    q = np.zeros((2, A.shape[1])); q[0, 0] = soft; q[0, 1] = -0.5 * soft; q[1, 1] = soft; q[1, 2] = -soft; sc_ = max(c[:EARLY].max(), 1e-12); Aw = np.vstack([Aw, q / sc_]); yw = np.concatenate([yw, [0.0, 0.0]])
                c = _nnls(Aw, yw)
        W[b] = c @ P
    if clean: W = clean_tail(W)
    if fill and mode == 'pred' and trust.any() and not trust.all(): W = fill_bands(W, trust)
    return W
def fill_bands(W, trust):
    """Bands where the sound has too little level to show its room (or where the room never shows by itself) take the
    room of the nearest bands that can be trusted: in between, a mix of the two neighbours in dB; beyond the ends, the
    last trusted band as it is."""
    W = W.copy(); idx = np.where(trust)[0]; L = 10 * np.log10(W + 1e-12)
    for b in np.where(~trust)[0]:
        lo = idx[idx < b]; hi = idx[idx > b]
        if len(lo) and len(hi): a = (b - lo[-1]) / (hi[0] - lo[-1]); W[b] = 10 ** (((1 - a) * L[lo[-1]] + a * L[hi[0]]) / 10.0)
        elif len(lo): W[b] = W[lo[-1]]
        else: W[b] = W[hi[0]]
    return W
def clean_tail(W, keep_db=20.0):
    """The fit leaves a faint floor at long lags (it costs little and pays for the wobble of real tails). The last part of
    the response's energy (what is left when the energy still to come has fallen keep_db under the whole) is not believed:
    from there on the curve continues at the slope the decay had on the way down, and never above what was fitted."""
    W = W.copy(); k = np.arange(K)
    for b in range(len(W)):
        w = W[b]; e = np.cumsum(w[::-1])[::-1]
        if e[0] <= 0: continue
        L = 10 * np.log10(e / e[0] + 1e-30); lo = np.where(L < -keep_db)[0]; hi = np.where(L < -keep_db / 2.0)[0]
        if len(lo) == 0 or len(hi) == 0: continue
        k2 = int(lo[0]); k1 = min(int(hi[0]), k2 - 1)
        if k1 < 0: continue
        sl = min((L[k2] - L[k1]) / max(k2 - k1, 1), -0.05)                                                               # dB per lag, never slower than 60 dB in 12.8 s
        ref = e[k2] * (1.0 - 10 ** (sl / 10.0)); W[b, k2:] = np.minimum(w[k2:], ref * 10 ** (sl * (k[k2:] - k2) / 10.0))
    return W
def numbers(W):
    """Per band: strength kappa (reverb energy per unit direct energy), decay time T60 (s) from the energy decay curve, and
    the decay curve itself in dB (bands, K)."""
    kappa = W.sum(1); edc = np.cumsum(W[:, ::-1], 1)[:, ::-1]; T60 = np.full(len(W), np.nan); L = 10 * np.log10(edc / np.maximum(edc[:, :1], 1e-30) + 1e-12)
    for b in range(len(W)):
        if kappa[b] <= 0: continue
        for (a0, a1) in ((-5.0, -20.0), (-3.0, -15.0), (-2.0, -10.0)):
            i0 = int(np.argmax(L[b] < a0)); i1 = int(np.argmax(L[b] < a1))
            if i1 > i0 + 2 and L[b].min() < a1:
                sl = np.polyfit(np.arange(i0, i1) * cm.DT, L[b, i0:i1], 1)[0]
                if sl < 0: T60[b] = -60.0 / sl; break
    return dict(kappa=kappa, T60=T60, edc_db=L)
def synth_ir(W, unshared=1.0, seed=1, length=None):
    """A stereo impulse response tail (n,2) whose band power follows W lag by lag (per channel: W times the power of a unit
    click), the two channels sharing a part so that the unshared share of the tail is `unshared` (0..1, scalar or per band)."""
    rng = np.random.default_rng(seed); kk = K if length is None else min(K, int(length / cm.DT) + 8); n = kk * cm.HOP + cm.NF; u = np.clip(np.broadcast_to(np.asarray(unshared, float), (cm.NB,)), 0.0, 1.0)
    com = rng.standard_normal(n); ind = rng.standard_normal((n, 2)); Xc = cm.fwd(np.stack([com, com], 1))[0]; Xi = cm.fwd(ind); idx = np.clip(np.searchsorted(cm.EDGES, np.arange(cm.NF // 2 + 1), side='right') - 1, 0, cm.NB - 1)
    ub = u[idx][None, :]; X = np.stack([np.sqrt(1 - ub) * Xc + np.sqrt(ub) * Xi[c] for c in range(2)])                # (2, T, bins): noise with the wanted left/right relation
    click = np.zeros((n, 2)); click[cm.NF] = 1.0; Pc = cm.bandpow(cm.fwd(click)).sum(0) / 2.0                            # band power of a unit click in one channel, summed over frames
    T = X.shape[1]; tgt = np.zeros((T, cm.NB)); k0 = (cm.NF + cm.NF) // cm.HOP                                          # frame index where a sound at sample NF is centred... lag 0
    # the click sits at sample NF of the padded picture: its power is spread over NF/HOP frames ending there; lag k of the response belongs k frames later
    c0 = int(np.argmax(cm.bandpow(cm.fwd(click)).sum(1))); L = min(kk, T - c0); tgt[c0:c0 + L] = W[:, :L].T
    P = (np.abs(X) ** 2) @ cm.BM.T                                                                                      # (2, T, bands) power of the noise as it is
    G = np.sqrt(tgt[None] * Pc[None, None, :] / np.maximum(P, 1e-30)); Gf = G[:, :, idx]; Gf[:, :, :cm.EDGES[0]] = Gf[:, :, cm.EDGES[0]:cm.EDGES[0] + 1]; Gf[:, :, cm.EDGES[-1]:] = 0.0
    h = cm.inv(X * Gf, n)
    # second pass: overlapping frames blur the shaping a little, so each band is brought to its exact energy (strength x a unit click's)
    Xh = cm.fwd(h); have = ((np.abs(Xh) ** 2) @ cm.BM.T).sum(1); want = W[:, :L].sum(1)[None, :] * Pc[None, :]; cf = np.sqrt(want / np.maximum(have, 1e-30))[:, idx]; cf[:, cm.EDGES[-1]:] = 0.0
    h = cm.inv(Xh * cf[:, None, :], n); h = h[cm.NF - cm.HOP:]                                                          # lag 0 then sits one frame step (10.7 ms) after the direct sound: the middle of where that energy really arrives
    return h
def apply_room(x, h, s=1.0):
    """x (n,2) plus s times x through the tail (each channel through its own tail)."""
    n = len(x); return x + s * np.stack([ss.fftconvolve(x[:, c], h[:, c])[:n] for c in range(2)], 1)
