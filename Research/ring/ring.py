# The ring engine (RING MOD; reference for Source/Core/Ring.h): a sound multiplied by a steady tone,
#   wet = dry * (a + b sin(wt + p)), the same on both sides.
# The gain is measured blind, in three steps:
#   1. candidates: the sound's power over time (left^2 + right^2), divided by its own local mean, carries lines at w
#      and at 2w that are a fixed share of its mean; the highest lines of its spectrum are tried, each as the carrier's
#      own line and as the line at twice the carrier;
#   2. first look, on little data: the power is sorted by where in the carrier's cycle it falls (64 steps) and by where
#      in the sound (8 stretches or more), and the gain that explains that table best is looked for (phase, a small
#      correction of the frequency, a / b). "Explains": the sound is taken as noise whose level is the gain (a
#      likelihood). The candidate that wins the most over "no gain at all" goes on; if none wins enough, there is none;
#   3. the same likelihood on the samples themselves, by Newton's method, with the floor under the gain (what the sound
#      holds where the gain is zero) fitted along with it; near the gain's zeros that pins the phase to a thousandth of
#      a sample.
# The check: the power sorted by the carrier's cycle must have the gain's own shape, the gain must dip by 10 dB or
# more (a shallower one is a tremolo), and the sound divided by the gain must not carry a steady offset (a plain note at
# the carrier's pitch is a "ring modulated" constant; that is not what is looked for).
# Taken out: the sound divided by the gain (with the floor where the gain goes through zero, and those few samples filled
# from their neighbours). Onto a second sound: multiplied by the gain.
import numpy as np
SR = 48000
FLO, FHI = 20.0, 5000.0          # carriers looked for
NB = 64                          # steps of the carrier's cycle
MOST = (6, 4)                    # lines of the power's spectrum that are tried: over SPLIT Hz, under it
THR = 0.2                        # .. and how high a line must be for that, against the power's mean
APART = 5.0                      # Hz: a peak nearer than this to a higher one belongs to it
SPLIT = 80.0                     # Hz: lines over this are looked for in the power levelled over W_HI samples, lines under it over W_LO
W_HI, W_LO = 1024, 8192
GATE = 1e-8                      # a sample counts if the sound around it is within 80 dB of its loudest moment: the tone sits in a quiet tail as it does in a loud stroke
GATE_LINES = 1e-3                # .. for the candidate lines: within 30 dB
SPEC = 1 << 19                   # the spectrum is taken over 10.9 s at most (the loudest)
LONG = 20 * SR                   # a longer sound: 20 stretches of one second, evenly spread, are measured
RHOS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8)        # first look: a / b tried
VS = (-0.4, -0.2, 0.0, 0.2, 0.4)             # first look: corrections of the frequency tried (radians at the two ends of the stretch looked at)
HOLE = 0.01                      # the samples where the gain is under this are not divided but filled from their neighbours
EPS0 = 0.05                      # first look: the floor under the gain
EPS_NC = 1e-3                    # the floor used when the gain never reaches zero
TWO_PI = 2 * np.pi

def power(y): return (y ** 2).sum(1)
def loudest(p, most, hop=SR // 2):
    """The stretch of `most` samples with the most power (looked for in steps of half a second): start, length."""
    n = len(p)
    if n <= most: return 0, n
    nb = n // hop; k = max(1, most // hop); c = np.concatenate([[0.0], np.cumsum(p[:nb * hop].reshape(nb, hop).sum(1))]); return min(int(np.argmax(c[k:nb + 1] - c[:nb + 1 - k])) * hop, n - most), most
def lines(pw, W, flo, fhi, most):
    """Candidate lines between flo and fhi Hz: the highest peaks in the spectrum of the power levelled over W samples
    (the loudest SPEC samples of it, and of those the ones within 30 dB of the loudest moment), measured against that
    power's mean. Returns their frequencies (Hz), the highest first."""
    a0, n = loudest(pw, SPEC)
    if n < 4096: return []
    pn, ok = local(pw, W, GATE_LINES); q = pn[a0:a0 + n]; k = ok[a0:a0 + n]
    if not k.any(): return []
    m = float(q[k].mean()); nfft = 1 << int(np.ceil(np.log2(2 * n))); win = 0.5 - 0.5 * np.cos(TWO_PI * np.arange(n) / n); P = np.abs(np.fft.rfft(np.where(k, q - m, 0.0) * win, nfft)); floor = THR * m * float(win[k].sum())
    lo = max(int(flo / SR * nfft), 2); hi = min(int(fhi / SR * nfft), len(P) - 2)
    if hi - lo < 3: return []
    seg = P[lo:hi]; cand = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] >= floor))[0] + 1; apart = APART / SR * nfft; idx = []
    for j in np.argsort(-seg[cand], kind='stable'):
        i0 = int(cand[j])
        if all(abs(i0 - u) > apart for u in idx): idx.append(i0)
        if len(idx) >= most: break
    out = []
    for i0 in idx:
        y0, y1, y2 = seg[i0 - 1], seg[i0], seg[i0 + 1]; den = y0 - 2 * y1 + y2; d = 0.5 * (y0 - y2) / den if den < 0 else 0.0; out.append(float((lo + i0 + d) * SR / nfft))
    return out
def spectrum_peaks(pw):
    """Candidate lines in the power's spectrum. A gain B (rho + sin) puts a line at w that is rho / (rho^2 + 1/2) of the
    power's mean and one at 2w that is 1/4 / (rho^2 + 1/2) of it, whatever the sound: for every rho up to 2 one of the
    two is 0.44 or more. So a line under THR of the mean is not tried, and of the others the highest are.
    The power is first divided by its own local mean. What the sound's loudness does in time (its beat, the shape of
    each stroke) sits around zero in the power's spectrum and, as a copy, around each of the tone's lines: on a drum
    loop or a row of short strokes that hides the lines among their neighbours and fills the low end with higher peaks
    than theirs. Divided by the local mean (21 ms) the loudness is gone and the lines stand alone. Lines under SPLIT Hz
    would go with it: they are looked for apart, with a local mean of 171 ms."""
    return lines(pw, W_HI, SPLIT, min(2 * FHI, 0.45 * SR), MOST[0]) + lines(pw, W_LO, FLO, SPLIT, MOST[1])
def samples(n):
    """The samples that are measured: all of them, or in a long sound 20 stretches of one second, evenly spread (the
    tone's pitch is pinned by how far apart the stretches are, not by how many samples there are)."""
    if n <= LONG: return np.arange(n)
    nc = n // SR; pick = (np.arange(20) * nc) // 20; return (pick[:, None] * SR + np.arange(SR)[None, :]).reshape(-1)
def window(f):
    """The length over which the sound's own level is taken for a carrier at f Hz: eight of its cycles, 21 ms at least
    (a power of two). For a slow tone (under 94 Hz) that would be 170 or 340 ms, longer than a drum stroke: there the
    length is a whole number of the tone's cycles instead (each of the two smoothings then leaves the tone's own cycle
    exactly as it is), 85 ms at least."""
    W8 = 1 << int(np.ceil(np.log2(max(8 * SR / f, 960.0)))); P = SR / f; return min(W8, 2 * int(np.floor(np.ceil(2048.0 / P) * P + 0.5)))
def local(pw, W, gate=None):
    """The power over its own local mean (a window of W samples, smoothed twice): what is left is the gain's cycle and
    chance, and a loud moment counts no more than a quiet one. Returns it and which samples count (the ones whose
    local mean is over gate times the highest one)."""
    n = len(pw); h = max(W // 2, 1); i = np.arange(n); a = np.maximum(i - h // 2, 0); b = np.minimum(i - h // 2 + h, n)
    e = pw
    for _ in range(2): c = np.concatenate([[0.0], np.cumsum(e)]); e = (c[b] - c[a]) / (b - a)
    ok = e > (GATE if gate is None else gate) * float(e.max()); return np.where(ok, pw / np.maximum(e, 1e-300), 0.0), ok

# ---------------------------------------------------------------- the first look: the power sorted by the carrier's cycle
MS = 8                           # first look: stretches along the part of the sound that is looked at
def fold(p, t, a0, n0, w, M=MS):
    """The table: per stretch (M of them along the samples a0 .. a0 + n0) and step of the carrier's cycle (NB) the
    power's sum and the number of samples. The cycle is counted from the middle of a0 .. a0 + n0: step k holds the
    phases k .. k + 1 (in 1 / NB of a turn) of w (t - middle)."""
    th = np.mod(w * (t - (a0 + 0.5 * n0)), TWO_PI); k = np.minimum((th * (NB / TWO_PI)).astype(np.int64), NB - 1); idx = ((t - a0) * M) // n0 * NB + k
    return np.bincount(idx, p, M * NB).reshape(M, NB), np.bincount(idx, None, M * NB).reshape(M, NB).astype(float)
def cell_gain(phi, v, rho, eps, M):
    """The gain's power (with the floor) averaged over each cell of the table, for the gain B (rho + sin(theta + phi + v tau)),
    tau = -1 .. 1 along the sound, B so that the mean power is 1."""
    d = TWO_PI / NB; tm = (2 * np.arange(M) + 1) / M - 1.0; ta = (np.arange(NB) * d)[None, :] + phi + v * tm[:, None]; tb = ta + d
    return (rho * rho + 0.5 + (2 * rho * (np.cos(ta) - np.cos(tb)) - 0.25 * (np.sin(2 * tb) - np.sin(2 * ta))) / d) / (rho * rho + 0.5) + eps * eps
def cell_cost(P, C, G):
    """Minus the likelihood of the table as noise at the level of the gain (per sample, up to a constant)."""
    N = float(C.sum()); return float((C * np.log(G)).sum() / N + np.log((P / G).sum() / N))
def golden(f, a, b, it=20):
    gr = 0.6180339887498949; c = b - gr * (b - a); d = a + gr * (b - a); fc, fd = f(c), f(d)
    for _ in range(it):
        if fc < fd: b, d, fd = d, c, fc; c = b - gr * (b - a); fc = f(c)
        else: a, c, fc = c, d, fd; d = a + gr * (b - a); fd = f(d)
    return 0.5 * (a + b)
def first_look(P, C):
    """The gain that explains the table best: phase (at the sound's middle), v, rho and the cost. A grid over all three,
    then each made finer in turn, three times."""
    M = len(P); N = float(C.sum()); best = None; ks = np.arange(NB)
    for v in VS:
        for rho in RHOS:
            G = cell_gain(0.0, v, rho, EPS0, M); L = np.log(G); R = 1.0 / G
            for j in range(NB):                           # the phase moved by j steps: the same table of gains, read j steps further on
                rl = (ks + j) % NB; J = float((C * L[:, rl]).sum() / N + np.log((P * R[:, rl]).sum() / N))
                if best is None or J < best[0]: best = (J, j * TWO_PI / NB, v, rho)
    J, phi, v, rho = best; d = TWO_PI / NB
    for _ in range(3):                                   # a / b may go under zero here: that is the same gain half a cycle on
        phi = golden(lambda x: cell_cost(P, C, cell_gain(x, v, rho, EPS0, M)), phi - d, phi + d)
        v = golden(lambda x: cell_cost(P, C, cell_gain(phi, x, rho, EPS0, M)), v - 0.2, v + 0.2)
        rho = golden(lambda x: cell_cost(P, C, cell_gain(phi, v, x, EPS0, M)), rho - 0.15, rho + 0.15)
    J = cell_cost(P, C, cell_gain(phi, v, rho, EPS0, M))
    if rho < 0: rho = -rho; phi += np.pi
    return float(np.mod(phi, TWO_PI)), v, rho, J

# ---------------------------------------------------------------- the likelihood on the samples
def passes(p, tau, tp, w, u):
    """One pass: the cost J at u = (phase at the middle, v, rho, log eps), its slope and its curvature (4 and 4 x 4), and
    the curvature the cost would have on average if the sound were exactly what the model takes it for (F: how the
    slopes of log (g^2 + eps^2) vary from sample to sample; never negative in any direction)."""
    phi, v, rho, e = u; eps2 = float(np.exp(2 * e)); th = w * tau + phi + v * tp; s = np.sin(th); c = np.cos(th); B2 = 1.0 / (rho * rho + 0.5); B = np.sqrt(B2)
    g = B * (rho + s); q = g * g + eps2; iq = 1.0 / q; r = p * iq; S = float(r.mean()); J = float(np.log(q).mean() + np.log(S)); Bc = B * c; Bs = B * s
    g1 = [Bc, Bc * tp, B - rho * B2 * g]; u_ = [2 * g * gi * iq for gi in g1] + [2 * eps2 * iq]
    g2 = {(0, 0): -Bs, (0, 1): -Bs * tp, (1, 1): -Bs * tp * tp, (0, 2): -rho * B2 * Bc, (1, 2): -rho * B2 * Bc * tp, (2, 2): -2 * rho * B2 * B - B2 * g + 3 * rho * rho * B2 * B2 * g}
    Ai = np.array([float(x.mean()) for x in u_]); Si = np.array([-float((r * x).mean()) for x in u_]); grad = Ai + Si / S; H = np.zeros((4, 4)); F = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            if j < 3: qij = (2 * g1[i] * g1[j] + 2 * g * g2[(i, j)]) * iq
            elif i == 3: qij = 4 * eps2 * iq
            else: qij = None
            uu = u_[i] * u_[j]; muu = float(uu.mean()); Aij = (float(qij.mean()) if qij is not None else 0.0) - muu; Sij = 2 * float((r * uu).mean()) - (float((r * qij).mean()) if qij is not None else 0.0)
            H[i, j] = H[j, i] = Aij + Sij / S - Si[i] * Si[j] / (S * S); F[i, j] = F[j, i] = muu - Ai[i] * Ai[j]
    return J, grad, H, F
E_LO, E_HI = float(np.log(1e-5)), float(np.log(0.3))
def newton(p, tau, tp, w, u, most=30, floor_free=True):
    """The cost made smallest from u on: Newton's steps, damped (a step is taken only if the cost comes down). Where the
    cost does not curve upwards in every direction (beside a zero of the gain that a sample sits on it curves downwards)
    the step is made with the average curvature F instead, which always points downhill.
    floor_free False: the floor stays where it is. Returns u, the cost, the passes made and the cost's slope in the floor."""
    u = np.array(u, float); lam = 1e-2; n_pass = 1; slope = 0.0
    def at(u):
        J, grad, H, F = passes(p, tau, tp, w, u); new = float(grad[3])
        if not floor_free:
            grad[3] = 0.0
            for M in (H, F): M[3, :] = 0.0; M[:, 3] = 0.0; M[3, 3] = 1.0
        try: np.linalg.cholesky(H)
        except np.linalg.LinAlgError: H = F
        return J, grad, H, new
    J, grad, H, slope = at(u)
    for _ in range(most):
        D = np.maximum(np.abs(np.diag(H)), 1e-12); step = None
        for _ in range(12):
            try: Lc = np.linalg.cholesky(H + lam * np.diag(D)); step = -np.linalg.solve(Lc.T, np.linalg.solve(Lc, grad)); break
            except np.linalg.LinAlgError: lam *= 10.0
        if step is None: break
        step = np.clip(step, [-0.3, -0.5, -0.3, -1.5], [0.3, 0.5, 0.3, 1.5]); un = u + step; un[3] = min(max(un[3], E_LO), E_HI)
        if bool(np.all(np.abs(un - u) < [1e-7, 1e-7, 1e-7, 1e-4])): break       # a step too small to matter: done
        Jn, gn, Hn, sn = at(un); n_pass += 1
        if Jn < J:
            small = J - Jn < 1e-12; u, J, grad, H, slope = un, Jn, gn, Hn, sn; lam = max(lam * 0.3, 1e-9)
            if small: break
        else:
            lam *= 10.0
            if lam > 1e12: break
    if u[2] < 0: u[2] = -u[2]; u[0] += np.pi              # the same gain half a cycle on
    return u, J, n_pass, slope
LADDER = tuple(0.05 * 10.0 ** (-0.5 * k) for k in range(8))       # the floors the fit comes down by before the floor is set free: 0.05, 0.016, 0.005 .. 1.6e-5
def settle(p, tau, tp, w, u):
    """The cost made smallest from u on, the floor coming down a step at a time and only then set free. Under a high
    floor the cost is smooth and its lowest point lies near the true one. Under a low floor every sample that sits next
    to a zero of the gain digs a narrow pit of its own, and where the tone's cycle meets the samples at the same places
    again and again (a round pitch) the pits add up to walls: a fit that lets the floor fall at once can end on the
    wrong side of one, a few thousandths of a radian from the tone. A step is left out when the sound wants no lower
    floor than the one before, or when the gain never reaches zero."""
    u = np.array(u, float); k = 0
    for fl in LADDER:
        u[3] = np.log(fl); u, J, kk, slope = newton(p, tau, tp, w, u, floor_free=False); k += kk
        if u[2] > 1.0 or slope <= 0.0: break
    u, J, kk, slope = newton(p, tau, tp, w, u); return u, J, k + kk
def gain(n, w, ph, rho, t0=0):
    """a + b sin(w t + p) with a / b = rho and mean power 1 (a^2 + b^2 / 2 = 1)."""
    b = 1.0 / np.sqrt(rho * rho + 0.5); return b * (rho + np.sin(w * (np.arange(n) + t0) + ph))
def profile(p, t, w, ph, rho, eps):
    """The check: the power sorted by the carrier's cycle (NB steps) against the gain's own shape. Returns how far the
    two are apart (dB, root mean square over the steps, each step counted by how much of the gain's power it holds)
    and how deep the gain's dip is (dB)."""
    th = np.mod(w * t + ph, TWO_PI); k = np.minimum((th * (NB / TWO_PI)).astype(np.int64), NB - 1); cnt = np.maximum(np.bincount(k, None, NB), 1); P = np.bincount(k, p, NB) / cnt
    b = 1.0 / np.sqrt(rho * rho + 0.5); g2 = (b * (rho + np.sin(w * t + ph))) ** 2 + eps * eps; G = np.bincount(k, g2, NB) / cnt; G = G / G.mean(); P = P / max(P.mean(), 1e-30)
    wgt = G / G.sum(); d = 10 * np.log10(np.maximum(P, 1e-9) / np.maximum(G, 1e-9)); d = d - float((wgt * d).sum())
    return float(np.sqrt((wgt * d * d).sum())), float(10 * np.log10(G.max() / max(G.min(), 1e-9)))
def ideal_win(rho):
    """What a sound multiplied by exactly this gain wins over "no gain at all" (per sample), with no floor."""
    return float(np.log(4 * rho * rho + 2) - (2 * np.log(rho + np.sqrt(rho * rho - 1)) if rho > 1 else 0.0))

# ---------------------------------------------------------------- taking it out
EPS_MIN = 1e-4                   # taking out: the gain is not taken to be known better than this
DOUBT = 100.0                    # a sample whose gain is under this many floors (and under GMAX) is not simply divided
GMAX = 0.1
KAPPA_MIN = 0.01                 # how much the sound's own course is trusted at most: its prediction error counts as 1 % of its power at least
ROUNDS = 3                       # the weights are set anew twice from what the first answer leaves unexplained
CZ = 3.0                         # .. a sample that is off by this many floors counts half
def divide(y, g, eps):
    """The sound divided by the gain, y g / (g^2 + eps^2): right wherever the gain is well over the floor."""
    return y * (g / (g * g + eps * eps))[:, None]
def take_out(y, g, eps, order=24, around=256):
    """The sound without the gain. Where the gain is far from zero the sound is divided by it. Next to a zero of the
    gain a sample says little (it is the sound times next to nothing, and the gain is known to the floor only), but the
    sound's own course says where it has to go: there each sample is the value that fits best with both, the sample
    over the gain counted by (gain / floor)^2 and the linear prediction of the sound around it (order 24, from 256
    samples on either side) counted by how well that sound can be predicted at all. Exactly at a zero that is the fill
    that makes the prediction error smallest; a few floors away it is the division. A sample that does not fit after
    that (the gain is off there by more than the floor) counts less in a second and a third round.
    Doubtful samples close together are worked out as one group (64 at most), every other doubtful sample in the
    stretch is unknown too, but gets its value from its own group, where it sits in the middle."""
    e = max(eps, EPS_MIN); x = divide(y, g, e); doubt = np.abs(g) < min(DOUBT * e, GMAX); idx = np.where(doubt)[0]; n = len(x)
    if not len(idx): return x
    out = x.copy(); q2 = (g / e) ** 2; i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] - idx[j] <= order and j - i < 63: j += 1          # nearer than the filter's length: one group (64 at most)
        a = max(idx[i] - around, 0); b = min(idx[j] + around + 1, n); m = doubt[a:b]; mi = np.where(m)[0]; gi = g[a:b][mi]; w0 = q2[a:b][mi]
        for ch in range(x.shape[1]):
            v = x[a:b, ch]; r = np.array([float(v[:len(v) - k] @ v[k:]) for k in range(order + 1)])        # the divided sound as it is: under the floor it is next to nothing by itself
            if not r[0] > 0: continue
            P = r[0] / len(v); r[0] *= 1.0 + 1e-6; c = np.zeros(order + 1); c[0] = 1.0; err = r[0]
            for q in range(1, order + 1):
                acc = r[q] + (float(c[1:q] @ r[q - 1:0:-1]) if q > 1 else 0.0); kk = -acc / err
                if q > 1: c[1:q] = c[1:q] + kk * c[q - 1:0:-1]
                c[q] = kk; err *= 1.0 - kk * kk
                if not err > 0: break
            kappa = min(max(err / r[0], KAPPA_MIN), 1.0) if err > 0 else KAPPA_MIN
            # |c * v|^2 + sum of kappa (g / e)^2 (v - y / g)^2 made smallest over the doubtful samples
            known = v.copy(); known[m] = 0.0; cc = np.correlate(c, c, 'full'); dlt = np.abs(mi[:, None] - mi[None, :]); A = np.where(dlt <= order, cc[order + np.minimum(dlt, order)], 0.0)
            rhs = -np.convolve(known, cc, 'same')[mi]; yi = y[a:b, ch][mi]; ridge = 1e-9 * A[0, 0]; wt = kappa * w0; u = None
            for it in range(ROUNDS):
                try: u = np.linalg.solve(A + np.diag(wt + ridge), rhs + wt * yi * gi / np.maximum(gi * gi, 1e-300))
                except np.linalg.LinAlgError: u = None; break
                if it + 1 < ROUNDS: wt = kappa * w0 / (1.0 + (yi - gi * u) ** 2 / (e * e * (P + u * u)) / (CZ * CZ))
            if u is not None: vv = v.copy(); vv[mi] = u; out[idx[i:j + 1], ch] = vv[idx[i:j + 1] - a]
        i = j + 1
    return out

# ---------------------------------------------------------------- the verdict
def find(y, rule=None):
    """dict(found, f Hz, rho = a / b, phase (at sample 0), eps, the numbers behind the verdict; x: the sound without the gain, g: the gain)."""
    R = dict(match=1.0, dip=10.0, offset=-20.0, win=0.3, first_win=0.2, holes=0.1); R.update(rule or {})
    n = len(y); pw = power(y); out = dict(found=False, n=n)
    if n < SR // 4 or not pw.sum() > 0: return out
    a0, n0 = loudest(pw, SPEC); mid = a0 + 0.5 * n0; tpart = np.arange(a0, a0 + n0); best = None; made = {}
    def levelled(W):                                       # the power over its local mean, and which samples count
        if W not in made: made[W] = local(pw, W)
        return made[W]
    def taken(W, t):
        pn, ok = levelled(W); t = t[ok[t]]; p = pn[t]; return p, t, (float(np.log(p.mean())) if len(t) else 0.0)      # the last: the cost of "no gain at all"
    for f0 in spectrum_peaks(pw):
        for k in (1, 2):                                   # the peak is the line at w, or the one at 2w
            f = f0 / k
            if not (FLO <= f <= FHI): continue
            W = window(f); p, t, flat = taken(W, tpart)
            if len(t) < SR // 8: continue
            w = TWO_PI * f / SR; P, C = fold(p, t, a0, n0, w); phi, v, rho, J = first_look(P, C)
            if best is None or flat - J > best[0]: best = (flat - J, w, phi, v, rho, W)
    if best is None: return out
    win0, w, phi, v, rho, W = best; out.update(first_win=win0, first_f=w * SR / TWO_PI, first_rho=rho, first_v=v)
    if win0 < R['first_win']: return out                   # the first look already says no
    def exact(t, half, u):                                 # the likelihood on the samples t, from u on; the frequency taken into w
        nonlocal w
        p, t, flat = taken(W, t); tau = t - mid; tp = tau / half; u, J, k1 = settle(p, tau, tp, w, u)
        if u[2] > 1.0:                                     # the gain never reaches zero: there is no floor to measure (it would only stand in for a / b), so none is used
            u[3] = np.log(EPS_NC); u, J, k2, _ = newton(p, tau, tp, w, u, floor_free=False); k1 += k2
        w = w + u[1] / half; u[1] = 0.0; return u, flat - J, k1, p, t
    u, win, n_pass, p, t = exact(tpart, 0.5 * n0, np.array([phi, v, rho, np.log(EPS0)]))
    if n > n0: u, win, k2, p, t = exact(samples(n), 0.5 * n, u); n_pass += k2      # a longer sound: now all of it, from what its loudest part gave
    ph = float(np.mod(u[0] - w * mid, TWO_PI)); rho = float(u[2]); eps = float(np.exp(u[3]))        # the phase at sample 0
    match, dip = profile(p, t, w, ph, rho, eps); g = gain(n, w, ph, rho); x = divide(y, g, eps)
    dc = float((x.mean(0) ** 2).sum() / max(float((x ** 2).mean(0).sum()), 1e-30)); hole = np.abs(g) < max(2 * eps, HOLE); holes = float(hole.mean())
    out.update(f=w * SR / TWO_PI, w=w, phase=ph, rho=rho, eps=eps, win=win, ideal=ideal_win(rho), match=match, dip=dip, offset=10 * np.log10(max(dc, 1e-12)), holes=holes, passes=n_pass)
    out['found'] = bool(match <= R['match'] and dip >= R['dip'] and out['offset'] <= R['offset'] and out['win'] >= R['win'] and holes <= R['holes'])
    if out['found']: out.update(x=take_out(y, g, eps), g=g)
    return out

# ---------------------------------------------------------------- the faders
def gain_at(f, s, n=None, carrier=1.0):
    """The gain with Amount at s: left of the middle a straight line from no gain (1) to the found one, right of it the
    tone part grows. carrier: the tone's pitch against the found one (1 = as found)."""
    n = n or f['n']; b = 1.0 / np.sqrt(f['rho'] ** 2 + 0.5); a = f['rho'] * b; sn = np.sin(carrier * f['w'] * np.arange(n) + f['phase'])
    return ((1.0 - s) + s * a if s <= 1.0 else a) + s * b * sn
def own(y, f, s, carrier=1.0):
    if not f.get('found') or (abs(s - 1.0) < 1e-12 and abs(carrier - 1.0) < 1e-12): return y.copy()
    return y + f['x'] * (gain_at(f, s, None, carrier) - f['g'])[:, None]
def onto(z, f, s, carrier=1.0):
    if not f.get('found') or s < 1e-12: return z.copy()
    return z * gain_at(f, s, len(z), carrier)[:, None]
