# Room capture and de-reverb, prototype. A room here is, per frequency band: how fast the tail dies (tau) and how strong
# the tail is next to the direct sound (A). Both are measured from the free decays found in the sound itself.
import numpy as np, scipy.signal as ss
SR = 48000
NF, HOP = 2048, 256                      # analysis frames: 43 ms window, 5.3 ms step
DT = HOP / SR
EDGES = 60.0 * (18000.0 / 60.0) ** (np.arange(21) / 20.0)      # 20 bands, 60 Hz .. 18 kHz
FC = np.sqrt(EDGES[:-1] * EDGES[1:])

def stft(x, nf=NF, hop=HOP, pad=True):
    """Frames of x. With pad, one window of silence goes in front and behind, so that a sound right at the start or end of
    the file is covered by as many frames as a sound in the middle (frame t is then centred on sample t*hop - nf/2)."""
    if pad: x = np.concatenate([np.zeros(nf), x, np.zeros(nf)])
    w = ss.get_window('hann', nf, fftbins=True); n = 1 + (len(x) - nf) // hop if len(x) >= nf else 0
    if n <= 0: return np.zeros((0, nf // 2 + 1), complex)
    idx = np.arange(nf)[None, :] + hop * np.arange(n)[:, None]; return np.fft.rfft(x[idx] * w, axis=1)

NF2, HOP2 = 512, 128                    # quick frames for transients: 10.7 ms window, 2.7 ms step
DT2 = HOP2 / SR
_bm = {}
def band_matrix(nf=NF):
    if nf in _bm: return _bm[nf]
    f = np.fft.rfftfreq(nf, 1 / SR); M = np.zeros((len(FC), len(f)))
    for b in range(len(FC)):
        lo, hi = EDGES[b], EDGES[b + 1]; m = (f >= lo) & (f < hi)
        if m.sum() < 2: m = np.zeros(len(f), bool); k = int(np.argmin(np.abs(f - FC[b]))); m[max(1, k - 1):k + 1] = True
        M[b, m] = 1.0
    _bm[nf] = M; return M

def band_power(x, nf=NF, hop=HOP):
    """x: (n,2). Power per band and frame, both channels added: (frames, bands)."""
    P = sum(np.abs(stft(x[:, c], nf, hop)) ** 2 for c in range(x.shape[1])); return P @ band_matrix(nf).T

def envelopes(x):
    return dict(slow=band_power(x, NF, HOP), fast=band_power(x, NF2, HOP2))

def _segments(L, floor):
    """Free decays in one band's level track L (dB): (start, end) frame pairs where the level keeps falling."""
    n = len(L); out = []; t = 3
    while t < n - 12:
        if L[t] > floor + 18 and L[t] >= L[max(0, t - 3):t + 4].max() - 1e-9:
            lo = L[t]; e = t + 1
            while e < n and e - t < int(4.0 / DT):
                if L[e] < lo: lo = L[e]
                if L[e] > lo + 4.0 or L[e] < floor + 3.0: break
                e += 1
            if e - t >= int(0.08 / DT) and L[t] - L[t:e].min() >= 12.0: out.append((t, e))
            t = e
        else: t += 1
    return out

def _late_fit(L, t0, t1):
    """One decay as a bent line: whatever falls first (the direct sound dropping away, the source's own decay), then the last
    straight stretch, which is the room if the decay got that far. Returns the last stretch's slope (dB/s), the knee frame,
    the misfit (dB rms) of the last stretch and its drop (dB)."""
    y = L[t0:t1]; n = len(y); x = np.arange(n) * DT
    def line(i0, i1):
        xs, ys = x[i0:i1], y[i0:i1]; s, c = np.polyfit(xs, ys, 1); return s, c, np.sqrt(np.mean((s * xs + c - ys) ** 2))
    minlen = int(0.08 / DT)
    if n < minlen + 2: return None
    best = None
    for k in list(range(2, n - minlen, max(1, (n - minlen) // 40))):
        s, c, err = line(k, n)
        if s >= 0: continue
        drop = -s * (n - k) * DT
        # the longest clean last stretch wins: misfit against length
        score = err - 2.5 * np.log10(max(drop, 1e-3)) - 1.0 * np.log10((n - k) * DT)
        if best is None or score < best[0]: best = (score, s, k, err, drop)
    if best is None: return None
    return best[1], best[2], best[3], best[4]

def analyse(x):
    """Measures the room on x (n,2). Returns dict with per-band tau (s), A (1/s), T60 (s), weight (evidence), and diagnostics."""
    return analyse_env(envelopes(x))

def analyse_env(env):
    E = env['slow']; nT, nB = E.shape
    k = np.ones(3) / 3; Es = np.stack([np.convolve(E[:, b], k, 'same') for b in range(nB)], 1)
    L = 10 * np.log10(Es + 1e-20)
    segs = [[] for _ in range(nB)]; tails = [np.zeros(nT, bool) for _ in range(nB)]; segs_all = []
    for b in range(nB):
        floor = max(np.percentile(L[:, b], 5), L[:, b].max() - 75.0)
        for (t0, t1) in _segments(L[:, b], floor):
            fit = _late_fit(L[:, b], t0, t1)
            if fit is None: continue
            s, kn, err, drop = fit
            if s > -8.0 or err > 2.5 or drop < 8.0: continue      # slower than a 7.5 s room, not a straight decay, or too shallow to tell
            span = (t1 - t0 - kn) * DT
            w = min(drop, 40.0) * min(span, 1.5) / (1.0 + err * err)
            segs[b].append((np.log(-s), w)); tails[b][t0 + kn:t1] = True; segs_all.append((b, t0, t1, kn, s, err, drop))
    # the room's decay rate per band: the value most decays agree on (this band and its neighbours), deep long decays counting most
    lr = np.full(nB, np.nan); wgt = np.zeros(nB); grid = np.linspace(np.log(8.0), np.log(600.0), 160)
    for b in range(nB):
        pool = [(v, w * (1.0 if bb == b else 0.5)) for bb in range(max(0, b - 1), min(nB, b + 2)) for (v, w) in segs[bb]]
        if not pool: continue
        v = np.array([p[0] for p in pool]); w = np.array([p[1] for p in pool]); dens = (w[None, :] * np.exp(-0.5 * ((grid[:, None] - v[None, :]) / 0.18) ** 2)).sum(1)
        lr[b] = grid[int(np.argmax(dens))]; wgt[b] = sum(w for (_, w) in segs[b])
    ok = np.isfinite(lr)
    if ok.sum() == 0: return dict(found=False, tau=np.zeros(nB), A=np.zeros(nB), T60=np.zeros(nB), weight=wgt, fc=FC, drr_db=np.full(nB, 99.0), segs=[])
    lr = np.interp(np.arange(nB), np.where(ok)[0], lr[ok]); lr = ss.medfilt(np.pad(lr, 1, 'edge'), 3)[1:-1]
    tau = 4.3429 / np.exp(lr)
    A = np.full(nB, np.nan)
    for b in range(nB):
        if tails[b].sum() >= 8: A[b] = _fit_level(E[:, b], tau[b], tails[b])
    okA = np.isfinite(A)
    if okA.sum() == 0: A = np.zeros(nB)
    else:
        la = np.interp(np.arange(nB), np.where(okA)[0], np.log(np.maximum(A[okA], 1e-6))); la = ss.medfilt(np.pad(la, 1, 'edge'), 3)[1:-1]; A = np.exp(la)
    return dict(found=True, tau=tau, A=A, T60=13.8155 * tau, weight=wgt, fc=FC, segs=segs_all, drr_db=-10 * np.log10(np.maximum(A * tau, 1e-9)))

import numba
@numba.njit(cache=True)
def _track(E, a, g):
    R = np.zeros(len(E)); r = 0.0
    for t in range(len(E)):
        R[t] = r; s = E[t] - r
        if s < 0.0: s = 0.0
        r = a * r + g * s
    return R
def reverb_track(E, tau, A, dt=DT):
    """The late-reverb power predicted for one band's power track E, and the direct part: E = S + R, R(t+1) = a R(t) + A dt S(t)."""
    return _track(np.ascontiguousarray(E, dtype=np.float64), float(np.exp(-dt / tau)), float(A * dt))

KAPPA_MAX = 30.0          # most room the fit will report: 15 dB more room than direct sound
def _fit_level(E, tau, tail):
    """The tail strength A: summed over the free decays, the predicted reverb power has to equal the measured power.
    (Power, not dB: a tonal sound rings a room unevenly from note to note, and the average ring is what the room is.)"""
    tot = E[tail].sum()
    def f(logA): return reverb_track(E, tau, np.exp(logA))[tail].sum() / max(tot, 1e-30)
    lo, hi = np.log(0.02), np.log(KAPPA_MAX / tau)
    if f(hi) < 1.0: return float(np.exp(hi))            # even the most room does not reach the measured tails: as wet as it gets
    if f(lo) > 1.0: return float(np.exp(lo))
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if f(mid) < 1.0: lo = mid
        else: hi = mid
    return float(np.exp(0.5 * (lo + hi)))

def contrast(Ef, tau):
    """Tail strength read from transients in one band's quick power track: after a hit, the level the hit leaves behind, next to
    the hit's own energy. Returns (logA per event, weight per event). Sustained sounds give values that are too high; the
    cleanest hits give the true one."""
    n = len(Ef); k3 = np.ones(3) / 3; Es = np.convolve(Ef, k3, 'same'); L = 10 * np.log10(Es + 1e-20); top = L.max()
    # onsets: the level jumps at least 6 dB above the 20 ms before it within 8 ms
    pre = np.convolve(Es, np.ones(8) / 8, 'full')[:n]                         # mean of frames t-7..t
    rise = np.full(n, -99.0); rise[11:] = L[11:] - 10 * np.log10(pre[:-11] + 1e-20)   # frame t against the mean ending at t-4... (t-11..t-4)
    ons = []; t = 12
    while t < n - 40:
        if rise[t] >= 6.0 and L[t] > top - 40.0 and rise[t] >= rise[t - 2:t + 3].max() - 1e-9:
            ons.append(t); t += 12
        else: t += 1
    if not ons: return np.zeros(0), np.zeros(0)
    grid = np.linspace(np.log(0.05), np.log(600.0), 40); a = np.exp(-DT2 / tau); out = []; wts = []
    Rg = np.stack([ss.lfilter([0.0, np.exp(g) * DT2], [1.0, -(a - np.exp(g) * DT2)], Ef) for g in grid])       # unclipped predicted reverb for every candidate strength
    Sg = Ef[None, :] - Rg
    for i, t0 in enumerate(ons):
        nxt = ons[i + 1] if i + 1 < len(ons) else n
        b0, b1 = t0 - 3, t0 + 6; p0, p1 = t0 + 7, min(t0 + 7 + 22, nxt - 4)       # the hit itself (window smear included), then up to 60 ms after it
        if p1 - p0 < 8 or b0 < 0: continue
        bump = Sg[:, b0:b1].sum(1); ped = Sg[:, p0:p1].mean(1) * (b1 - b0)
        ratio = ped / np.maximum(bump, 1e-20)                                    # what is left over after the hit, per unit of hit energy; falls as A rises
        if ratio[0] <= 0: out.append(grid[0]); wts.append(min(rise[t0], 20.0)); continue
        j = np.where(ratio <= 0)[0]
        if len(j) == 0: continue                                                # still sounding at every strength: a held note, no evidence
        j = j[0]; r0, r1 = ratio[j - 1], ratio[j]; g = grid[j - 1] + (grid[j] - grid[j - 1]) * r0 / max(r0 - r1, 1e-12)
        out.append(g); wts.append(min(rise[t0], 20.0))
    return np.array(out), np.array(wts)


# ---------------------------------------------------------------- stereo: what both channels share, and what they do not
def diffuse_power(x, nf=NF, hop=HOP, tavg=0.128, favg=2):
    """x: (n,2). Per band and frame: total power E and the power D that the two channels do NOT share. For every frequency
    bin the 2x2 left/right covariance is averaged over tavg seconds and +-favg bins; a single source at a fixed place in the
    stereo picture (panned, or delayed between the channels) gives a covariance of rank one however it changes in time, and
    anything left over (the smaller eigenvalue, in both channels) is sound that arrives differently every moment: a room's
    tail, or a source that is wide by itself. Returns (E, D), each (frames, bands)."""
    L = stft(x[:, 0], nf, hop); R = stft(x[:, 1], nf, hop)
    pl = np.abs(L) ** 2; pr = np.abs(R) ** 2; c = L * np.conj(R)
    k = max(1, int(round(tavg / (hop / SR)))); kt = np.ones(k) / k; kf = np.ones(2 * favg + 1) / (2 * favg + 1)
    def sm(a):
        a = ss.fftconvolve(a, kt[:, None], 'same'); return ss.fftconvolve(a, kf[None, :], 'same')
    pls, prs, cs = sm(pl), sm(pr), sm(c.real) + 1j * sm(c.imag)
    lam = 0.5 * ((pls + prs) - np.sqrt(np.maximum((pls - prs) ** 2 + 4 * np.abs(cs) ** 2, 0.0)))      # the smaller eigenvalue
    M = band_matrix(nf).T; return (pls + prs) @ M, np.maximum(2.0 * lam, 0.0) @ M
