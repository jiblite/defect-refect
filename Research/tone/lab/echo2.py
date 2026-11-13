# Echo finder, second version. A short reflection (comb filter) puts the ripple g*cos(2 pi f d) on the log spectrum of EVERY frame.
# Candidates: peaks in the cepstrum of the long-term log spectrum. Check, frame by frame: the size of that ripple in the frame's own
# spectrum (only the bins where the frame has sound; the slow trend taken out with a running mean one ripple period long). A reflection
# gives about g in every frame, a played note only while it sounds: the median over the frames decides. A steady pitch is told apart by
# its second peak: positive at twice the delay, where a reflection has -g^2/2.
import numpy as np, scipy.ndimage as ndi
import feat
SR = 48000; N = feat.N; FB = feat.FB; QM = 1441; NB = len(FB)
_band = (FB >= 100) & (FB <= 16000); W0 = np.zeros(NB); W0[_band] = np.hanning(_band.sum())
QLO, QHI = int(0.5 * 48), int(15 * 48)
def frame_logs(Q, range_db=50.0):
    """Q: power (T, bins). ln-magnitude per frame and which bins count in each frame (within range_db of the frame's loudest bin, 100 Hz..16 kHz)."""
    mx = Q[:, _band].max(1, keepdims=True); V = (Q > mx * 10 ** (-range_db / 10)) & _band[None, :]; L = 0.5 * np.log(np.maximum(Q, 1e-30)); return L, V
def ripple(L, V, d):
    """Per frame: gain of cos(2 pi f d / SR) in the frame's log spectrum (None where the frame has too little to say). d in samples."""
    per = max(3, int(round(N / d))); Vf = V.astype(float)
    num = ndi.uniform_filter1d(L * Vf, per, axis=1, mode='constant'); den = ndi.uniform_filter1d(Vf, per, axis=1, mode='constant'); R = (L - num / np.maximum(den, 1e-9)) * Vf
    c = np.cos(2 * np.pi * FB * d / SR); a = (R * c[None, :]).sum(1) / np.maximum((Vf * c[None, :] ** 2).sum(1), 1e-9); ok = Vf.sum(1) >= max(60, 3 * per)
    return a, ok
def long_term(L, V):
    """Mean log spectrum over the frames where each bin counts, and how many frames that were (fraction)."""
    Vf = V.astype(float); n = Vf.sum(0); return (L * Vf).sum(0) / np.maximum(n, 1), n / len(L)
def cepstrum(Lm, w):
    x = (Lm - (Lm * w).sum() / max(w.sum(), 1e-9)) * w; return np.fft.irfft(x, N)[:QM] * (N / max(w.sum(), 1e-9))
def project(Lm, w, d):
    d = np.atleast_1d(d).astype(float); x = (Lm - (Lm * w).sum() / max(w.sum(), 1e-9)) * w; return (2.0 / max(w.sum(), 1e-9)) * (np.cos(2 * np.pi * np.outer(d, np.arange(NB)) / N) @ x)
def comb_ln(d, g): return 0.5 * np.log(np.maximum(1 + g * g + 2 * g * np.cos(2 * np.pi * FB * d / SR), 1e-12))
def local_valid(Q, d, range_db=20.0):
    """Bins that carry a frame's own sound rather than leakage from a stronger neighbour: within range_db of the largest bin nearby
    (the neighbourhood is one ripple period wide, at least 32 bins)."""
    per = max(32, int(round(N / d))); mx = ndi.maximum_filter1d(Q, per, axis=1, mode='nearest'); return (Q > mx * 10 ** (-range_db / 10)) & _band[None, :] & (Q > Q[:, _band].max(1, keepdims=True) * 1e-6)
def lmean_ln(Q, clamp_db=40.0):
    """ln-magnitude long-term spectrum: mean over frames of the log power, each bin held at clamp_db under its own loud frames; and the loud level in dB."""
    top = np.percentile(Q, 95, axis=0); return 0.5 * np.log(np.maximum(Q, np.maximum(top * 10 ** (-clamp_db / 10), 1e-30))).mean(0), 10 * np.log10(np.maximum(top, 1e-30))
def weight(lev):
    """Bins that count: 100 Hz..16 kHz with a Hann taper, and only where the sound has something. The level is smoothed over 1.5 kHz first, so that
    the comb's own notches do not print themselves into the weights (that would smear the cepstral peak into side peaks)."""
    sm = ndi.uniform_filter1d(np.maximum(lev, lev[_band].max() - 80.0), 257, mode='nearest'); return W0 * np.clip((sm - (sm[_band].max() - 60.0)) / 10.0, 0, 1)
def thr_default(d): return 0.15 * np.ones_like(np.atleast_1d(d), float)
def find(Q, thr=thr_default, max_echoes=3, ncand=12, frac2=0.2, agree_min=0.8, verbose=False, collect=None):
    """Q: power (T, bins) of the frames that count. Returns [(delay in samples, gain)], strongest first. thr: function delay -> smallest gain that counts."""
    Lm, lev = lmean_ln(Q); w = weight(lev); out = []; Lf = 0.5 * np.log(np.maximum(Q, 1e-30))
    if w.sum() < 50: return out
    for _ in range(max_echoes):
        c = cepstrum(Lm, w); q = np.arange(QM); r = np.abs(c) / thr(q); r[:QLO] = 0; r[QHI:] = 0; best = None
        for _c in range(ncand):
            k = int(np.argmax(r))
            if r[k] < 0.7: break
            r[max(0, k - 3):k + 4] = 0
            dd = k + np.arange(-1.5, 1.5001, 0.05); a = project(Lm, w, dd); j = int(np.argmax(np.abs(a))); d = float(dd[j]); g = float(a[j]); g2 = float(project(Lm, w, 2 * d)[0])
            V = local_valid(Q, d); at, ok = ripple(Lf, V, d); agree = float((np.sign(at[ok]) == np.sign(g)).mean()) if ok.sum() >= 8 else 0.0; med = float(np.median(at[ok])) if ok.sum() >= 8 else 0.0
            if verbose: print('   cand d %.2f smp (%.3f ms): gain %+.3f, second peak %+.3f (a reflection: %+.3f), frames agreeing %.0f%% (median ripple per frame %+.3f)' % (d, d / 48, g, g2, -0.5 * g * g, 100 * agree, med))
            if collect is not None:
                T = len(Q); parts = []
                for nseg in (3, 6):
                    for i_ in range(nseg):
                        Lp, lp = lmean_ln(Q[i_ * T // nseg:(i_ + 1) * T // nseg]); wp = weight(lp); parts.append(float(project(Lp, wp, d)[0]) if wp.sum() > 50 else 0.0)
                collect.append((d, g, g2, agree, med) + tuple(parts))
            if abs(g) >= thr(d)[0] and agree >= agree_min and g2 <= -frac2 * g * g and (best is None or abs(g) / thr(d)[0] > abs(best[1]) / thr(best[0])[0]): best = (d, g)
        if best is None: break
        d, g = best; g = float(np.clip(g, -0.95, 0.95)); out.append((d, g)); cl = comb_ln(d, g); Lm = Lm - cl; Lf = Lf - cl[None, :]; Q = Q * np.exp(-2 * cl)[None, :]
    return out
def find3(Q, sens=1.0, max_echoes=3, ncand=40, verbose=False):
    """Final rule. A candidate (cepstral peak of the whole sound's long-term log spectrum, 0.5..15 ms) is a reflection when
    its gain is at least 0.20 (0.30 for delays under 1 ms), divided by sens; its second peak, at twice the delay, is negative (at most
    -0.1 g^2; a reflection has -0.5 g^2, a steady pitch a positive one); and every sixth of the sound (in time, each with its own
    long-term spectrum) shows it with the same sign and at least 0.4 of its size. The gain reported is the median of the whole and the sixths."""
    T = len(Q); nseg = min(6, T // 12)
    if nseg < 2: return []
    parts = [Q] + [Q[i * T // nseg:(i + 1) * T // nseg] for i in range(nseg)]; LM = []; WW = []
    for p in parts: Lm, lev = lmean_ln(p); LM.append(Lm); WW.append(weight(lev))
    if WW[0].sum() < 50: return []
    thr = lambda d: (0.30 if d < 48 else 0.20) / sens; tq = np.where(np.arange(QM) < 48, 0.30, 0.20) / sens; out = []
    for _ in range(max_echoes):
        a = np.abs(cepstrum(LM[0], WW[0])) / tq; a[:QLO] = 0; a[QHI:] = 0; best = None
        for _c in range(ncand):
            k = int(np.argmax(a))
            if a[k] < 0.8: break
            a[max(0, k - 3):k + 4] = 0
            dd = k + np.arange(-1.5, 1.5001, 0.05); pr = project(LM[0], WW[0], dd); j = int(np.argmax(np.abs(pr))); d = float(dd[j]); g = float(pr[j])
            if abs(g) < thr(d): continue
            g2 = float(project(LM[0], WW[0], 2 * d)[0]); gs = [float(project(LM[i], WW[i], d)[0]) if WW[i].sum() >= 50 else 0.0 for i in range(1, nseg + 1)]
            ok = g2 <= -0.1 * g * g and all(np.sign(x) == np.sign(g) and abs(x) >= 0.4 * abs(g) for x in gs)
            if verbose: print('   cand d %.2f smp (%.3f ms): gain %+.3f, second peak %+.3f (a reflection: %+.3f), sixths %s -> %s' % (d, d / 48, g, g2, -0.5 * g * g, ' '.join('%+.2f' % x for x in gs), 'yes' if ok else 'no'))
            if ok and (best is None or abs(g) / thr(d) > best[2]): best = (d, float(np.median([g] + gs)), abs(g) / thr(d))
        if best is None: break
        d, g = best[0], float(np.clip(best[1], -0.95, 0.95)); out.append((d, g)); cl = comb_ln(d, g); LM = [L - cl for L in LM]
    return out
def refine(Lm, w, d, g, span=1.0, step=0.02, K=4):
    """Delay and gain by a least-squares fit of the whole comb curve ln|1 + g e^{-j theta}| = sum_k (-1)^(k+1) g^k/k cos(k theta) (its first
    K terms), on a grid of delays around d and gains around g."""
    dd = d + np.arange(-span, span + 1e-9, step); x = (Lm - (Lm * w).sum() / max(w.sum(), 1e-9)) * w; kb = np.arange(NB)
    P = np.stack([(2.0 / max(w.sum(), 1e-9)) * (np.cos(2 * np.pi * np.outer(k * dd, kb) / N) @ x) for k in range(1, K + 1)])       # (K, delays): gain of cos(k theta)
    best = None
    for gg in g * np.arange(0.7, 1.31, 0.05):
        co = np.array([(-1.0) ** (k + 1) * gg ** k / k for k in range(1, K + 1)]); sc = -2 * (co[:, None] * P).sum(0) + (co ** 2).sum(); j = int(np.argmin(sc))
        if best is None or sc[j] < best[0]: best = (float(sc[j]), float(dd[j]), float(gg))
    return best[1], best[2]
def find4(Q, sens=1.0, max_echoes=3, ncand=40, verbose=False, fit=True):
    """As find3, with: parts of the sound that have too little to say are left out (at least 3 parts must speak), and delay and gain are
    refined by a fit of the whole comb curve."""
    T = len(Q); nseg = min(6, T // 8)
    if nseg < 3: return []
    parts = [Q] + [Q[i * T // nseg:(i + 1) * T // nseg] for i in range(nseg)]; LM = []; WW = []
    for p in parts: Lm, lev = lmean_ln(p); LM.append(Lm); WW.append(weight(lev))
    if WW[0].sum() < 50: return []
    use = [i for i in range(1, nseg + 1) if WW[i].sum() >= 0.5 * WW[0].sum()]
    if len(use) < 3: return []
    thr = lambda d: (0.30 if d < 48 else 0.20) / sens; tq = np.where(np.arange(QM) < 48, 0.30, 0.20) / sens; out = []
    for _ in range(max_echoes):
        a = np.abs(cepstrum(LM[0], WW[0])) / tq; a[:QLO] = 0; a[QHI:] = 0; best = None
        for _c in range(ncand):
            k = int(np.argmax(a))
            if a[k] < 0.8: break
            a[max(0, k - 3):k + 4] = 0
            dd = k + np.arange(-1.5, 1.5001, 0.05); pr = project(LM[0], WW[0], dd); j = int(np.argmax(np.abs(pr))); d = float(dd[j]); g = float(pr[j])
            if abs(g) < thr(d): continue
            g2 = float(project(LM[0], WW[0], 2 * d)[0]); gs = [float(project(LM[i], WW[i], d)[0]) for i in use]
            ok = g2 <= -0.1 * g * g and all(np.sign(x) == np.sign(g) and abs(x) >= 0.4 * abs(g) for x in gs)
            if verbose: print('   cand d %.2f smp (%.3f ms): gain %+.3f, second peak %+.3f (a reflection: %+.3f), parts %s -> %s' % (d, d / 48, g, g2, -0.5 * g * g, ' '.join('%+.2f' % x for x in gs), 'yes' if ok else 'no'))
            if ok and (best is None or abs(g) / thr(d) > best[2]): best = (d, float(np.median([g] + gs)), abs(g) / thr(d))
        if best is None: break
        d, g = best[0], best[1]
        if fit: d, g = refine(LM[0], WW[0], d, g)
        g = float(np.clip(g, -0.95, 0.95)); out.append((d, g)); cl = comb_ln(d, g); LM = [L - cl for L in LM]
    return out
def find5(Q, sens=1.0, max_echoes=2, ncand=40, iso=0.5, iso_span=10, verbose=False, tmin=0.25, tlow=0.40):
    """As find4, plus: a reflection stands alone. After its comb is taken out, nothing of half its size may be left in the cepstrum within
    +-iso_span samples of its delay (a cluster of peaks is a formant-like ripple of the sound itself, not a reflection)."""
    T = len(Q); nseg = min(6, T // 8)
    if nseg < 3: return []
    parts = [Q] + [Q[i * T // nseg:(i + 1) * T // nseg] for i in range(nseg)]; LM = []; WW = []
    for p in parts: Lm, lev = lmean_ln(p); LM.append(Lm); WW.append(weight(lev))
    if WW[0].sum() < 50: return []
    use = [i for i in range(1, nseg + 1) if WW[i].sum() >= 0.5 * WW[0].sum()]
    if len(use) < 3: return []
    thr = lambda d: (tlow if d < 48 else tmin) / sens; tq = np.where(np.arange(QM) < 48, tlow, tmin) / sens; out = []
    for _ in range(max_echoes):
        a = np.abs(cepstrum(LM[0], WW[0])) / tq; a[:QLO] = 0; a[QHI:] = 0; cands = []
        for _c in range(ncand):
            k = int(np.argmax(a))
            if a[k] < 0.8: break
            a[max(0, k - 3):k + 4] = 0
            dd = k + np.arange(-1.5, 1.5001, 0.05); pr = project(LM[0], WW[0], dd); j = int(np.argmax(np.abs(pr))); d = float(dd[j]); g = float(pr[j])
            if abs(g) < thr(d): continue
            if float(project(LM[0], WW[0], 2 * d)[0]) > -0.1 * g * g: continue
            gs = [float(project(LM[i], WW[i], d)[0]) for i in use]
            if not all(np.sign(x) == np.sign(g) and abs(x) >= 0.4 * abs(g) for x in gs): continue
            cands.append((abs(g) / thr(d), d, float(np.median([g] + gs))))
        got = None
        for _, d, g in sorted(cands, reverse=True):
            d, g = refine(LM[0], WW[0], d, g); g = float(np.clip(g, -0.95, 0.95)); cl = comb_ln(d, g); res = cepstrum(LM[0] - cl, WW[0]); k = int(round(d)); left = float(np.abs(res[max(QLO, k - iso_span):k + iso_span + 1]).max())
            if verbose: print('   d %.2f g %+.3f: left over near the delay %.3f (limit %.3f) -> %s' % (d, g, left, iso * abs(g), 'yes' if left <= iso * abs(g) else 'no'))
            if left <= iso * abs(g): got = (d, g, cl); break
        if got is None: break
        out.append(got[:2]); LM = [L - got[2] for L in LM]
    return out
