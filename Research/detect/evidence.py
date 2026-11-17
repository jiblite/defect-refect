# Physical evidence for effects, measured on a whole stretch of sound (not on 3 s windows): the numbers the verification
# step asks for before an effect is shown. Plain numpy / scipy; Source/Core/Verify.h evidence() is the same thing in C++
# (Tools/verify_parity.py compares the two number by number).
#   stereo:    is one side a late copy of the other (Haas), is left/right one source through a fixed path (width, pan)
#   level:     a periodic level pattern, and whether all bands follow it in parallel (tremolo, gate); left against right (auto pan)
#   samples:   few distinct sample values, held samples (bit crush), flat tops (clipping)
#   carrier:   a steady line in the signal's envelope at audio rate (ring modulator)
#   repeats:   stretches that repeat exactly (stutter)
#   spectrum:  the long-term spectrum's edges (a fixed filter) and its top end (exciter); a ceiling on the loud parts
# The run that made the app's models also measured 29 further numbers (an older check for comb teeth and sweeps, the movement
# of the spectrum's shape and of its roll-off, echoes of onsets). The models in the app do not read them (verifier.py DROP),
# so they are not here and not in the C++: on the held-out sounds they bought 2 points of effects found (48 % against 46 %)
# and nothing on the clean sounds (92 % against 93 % come out clean).
import numpy as np, scipy.signal as ss
from scipy.ndimage import uniform_filter1d, median_filter, maximum_filter1d
SR = 48000; FPS = 100.0
def ceps(x):
    """Real cepstrum of a whole channel (Hand.h ceps)."""
    n = 1 << int(np.ceil(np.log2(len(x)))); X = np.fft.rfft(x * np.hanning(len(x)), n); return np.fft.irfft(np.log(np.abs(X) + 1e-6 * np.abs(X).max()))

def _line(v, lo, hi, fs, w=None):
    """Strongest line of a track between lo and hi Hz: (rate, share of the band's power that sits in the line, r2 of a sine fit, amplitude, spectrum, freqs)."""
    v = np.asarray(v, float); n = len(v)
    if n < 16 or np.std(v) < 1e-9: return 0.0, 0.0, 0.0, 0.0, None, None
    v = v - v.mean(); N = 1 << int(np.ceil(np.log2(n * 8))); F = np.fft.rfft(v * np.hanning(n), N); f = np.fft.rfftfreq(N, 1 / fs); P = np.abs(F) ** 2
    sel = (f >= max(lo, 1.5 * fs / n)) & (f <= hi)
    if not sel.any(): return 0.0, 0.0, 0.0, 0.0, F, f
    k = int(np.argmax(np.where(sel, P, 0))); r = f[k]; half = 2.0 * fs / n                       # a Hann line is 2 bins of the unpadded transform wide either way
    share = P[sel & (np.abs(f - r) <= half)].sum() / (P[sel].sum() + 1e-30)
    t = np.arange(n) / fs; A = np.stack([np.sin(2 * np.pi * r * t), np.cos(2 * np.pi * r * t), np.ones(n)], 1); c = np.linalg.lstsq(A, v, rcond=None)[0]
    r2 = max(0.0, 1 - np.var(v - A @ c) / (np.var(v) + 1e-30))
    return float(r), float(share), float(r2), float(np.hypot(c[0], c[1])), F, f
def _amp_at(F, f, r): return np.abs(F[int(np.argmin(np.abs(f - r)))]) if F is not None else 0.0
def _camp(E, r, fs):
    """Complex amplitude of each row of E (bands x frames) at rate r."""
    n = E.shape[1]; t = np.arange(n) / fs; return (E * np.hanning(n)) @ np.exp(-2j * np.pi * r * t)

NAMES = []
def evidence(y):
    """y: (n, 2) at 48 kHz -> dict of 55 numbers (always the same keys, NAMES)."""
    o = {}; L, R = y[:, 0], y[:, 1]; M = 0.5 * (L + R); S = 0.5 * (L - R); n = len(M); pk = np.abs(y).max() + 1e-12
    # ---------------------------------------------------------------- stereo
    eM = (M ** 2).mean() + 1e-20; eS = (S ** 2).mean() + 1e-20; o['st_side_db'] = float(np.clip(10 * np.log10(eS / eM), -80, 20))
    sl, sr = L.std(), R.std(); wide = sl > 1e-7 and sr > 1e-7 and o['st_side_db'] > -60
    o.update(st_corr0=1.0, st_xc_peak=1.0, st_xc_lag_ms=0.0, st_xc_gain=0.0, st_xc_neg=0.0, st_msc=1.0, st_msc_hi=1.0, st_bal_db=0.0)
    if wide:
        o['st_bal_db'] = float(20 * np.log10(sl / sr)); m = min(n, 12 * SR); a = L[:m] - L[:m].mean(); b = R[:m] - R[:m].mean(); N = 1 << int(np.ceil(np.log2(2 * m)))
        cc = np.fft.irfft(np.fft.rfft(a, N) * np.conj(np.fft.rfft(b, N)), N) / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-30)
        K = int(0.04 * SR); lags = np.r_[np.arange(0, K + 1), np.arange(-K, 0)]; v = np.r_[cc[:K + 1], cc[-K:]]
        far = np.abs(lags) >= int(0.002 * SR); k = int(np.argmax(np.where(far, v, -2)))                 # a late copy: at least 2 ms away (closer is how microphones stand)
        o['st_corr0'] = float(v[0]); o['st_xc_peak'] = float(v[k]); o['st_xc_lag_ms'] = float(abs(lags[k]) / SR * 1e3); o['st_xc_gain'] = float(v[k] - np.abs(v[np.abs(lags) < int(0.0005 * SR)]).max())
        o['st_xc_neg'] = float(-np.where(far, v, 2).min())
        f, Pxy = ss.csd(L, R, SR, nperseg=2048, noverlap=1024); _, Pxx = ss.welch(L, SR, nperseg=2048, noverlap=1024); _, Pyy = ss.welch(R, SR, nperseg=2048, noverlap=1024)
        msc = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-30); w = np.sqrt(Pxx * Pyy); band = (f > 80) & (f < 16000); hi = (f > 1500) & (f < 16000)
        o['st_msc'] = float((msc * w)[band].sum() / (w[band].sum() + 1e-30)); o['st_msc_hi'] = float((msc * w)[hi].sum() / (w[hi].sum() + 1e-30))
    # ---------------------------------------------------------------- band levels over time (100 frames a second, 7 octave bands 100 Hz - 12.8 kHz)
    f, _, ZL = ss.stft(L, SR, 'hann', 1024, 1024 - 480, boundary=None, padded=False); _, _, ZR = ss.stft(R, SR, 'hann', 1024, 1024 - 480, boundary=None, padded=False)
    PM = np.abs(0.5 * (ZL + ZR)) ** 2; PL = np.abs(ZL) ** 2; PR = np.abs(ZR) ** 2; T = PM.shape[1]
    edges = 100.0 * 2 ** np.arange(8); bands = [(f >= edges[i]) & (f < edges[i + 1]) for i in range(7)]
    def blev(P): return np.array([10 * np.log10(P[b].sum(0) + 1e-14) for b in bands])
    EM = blev(PM); EL = blev(PL); ER = blev(PR); tot = 10 * np.log10(PM[(f >= 100) & (f < 12800)].sum(0) + 1e-14); top = tot.max()
    act = tot > top - 40; okb = EM.max(1) > EM.max() - 40; nb = int(okb.sum())
    span = np.flatnonzero(act); t0, t1 = (int(span[0]), int(span[-1]) + 1) if len(span) else (0, T)
    o.update(lv_rate=0.0, lv_share=0.0, lv_r2=0.0, lv_depth=0.0, lv_par=0.0, lv_ampcv=1.0, lv_h2=0.0, lv_h3=0.0, lv_dip=0.0, lv_edge=0.0, lv_halves=1.0, lv_sd=0.0)
    if T >= 300 and nb >= 2 and act.sum() > 100:
        E = np.maximum(EM[okb], EM[okb].max(1, keepdims=True) - 60); D = (E - uniform_filter1d(E, 100, axis=1, mode='nearest')) * act
        c = D.mean(0); r, share, r2, amp, F, ff = _line(c, 1.0, 16.0, FPS); o.update(lv_rate=r, lv_share=share, lv_r2=r2, lv_depth=2 * amp, lv_sd=float(c[act].std()))
        if r > 0:
            A = _camp(D, r, FPS); o['lv_par'] = float(np.abs(A.sum()) ** 2 / (nb * (np.abs(A) ** 2).sum() + 1e-30)); o['lv_ampcv'] = float(np.abs(A).std() / (np.abs(A).mean() + 1e-30))
            a1 = _amp_at(F, ff, r) + 1e-30; o['lv_h2'] = float(_amp_at(F, ff, 2 * r) / a1); o['lv_h3'] = float(_amp_at(F, ff, 3 * r) / a1)
            h = len(c) // 2; ra = _line(c[:h], 1.0, 16.0, FPS)[0]; rb = _line(c[h:], 1.0, 16.0, FPS)[0]; o['lv_halves'] = float(abs(np.log2((ra + 1e-3) / (rb + 1e-3))))      # the same speed in both halves?
        seg = tot[t0:t1]; o['lv_dip'] = float((seg < maximum_filter1d(seg, 100, mode='nearest') - 15).mean()); o['lv_edge'] = float(np.percentile(np.abs(np.diff(seg)), 90)) if len(seg) > 2 else 0.0
    # ---------------------------------------------------------------- left against right over time
    o.update(pn_rate=0.0, pn_share=0.0, pn_r2=0.0, pn_depth=0.0, pn_par=0.0, pn_sd=0.0, pn_envcorr=1.0)
    if wide and T >= 300 and nb >= 2 and act.sum() > 100:
        I = (np.maximum(EL[okb], EM[okb].max() - 70) - np.maximum(ER[okb], EM[okb].max() - 70)) * act; I = I - (I.sum(1, keepdims=True) / act.sum())* act
        d = I.mean(0); r, share, r2, amp, F, ff = _line(median_filter(d, 5), 0.1, 6.0, FPS); o.update(pn_rate=r, pn_share=share, pn_r2=r2, pn_depth=2 * amp, pn_sd=float(d[act].std()))
        if r > 0: A = _camp(I, r, FPS); o['pn_par'] = float(np.abs(A.sum()) ** 2 / (nb * (np.abs(A) ** 2).sum() + 1e-30))
        tl = 10 * np.log10(PL[(f >= 100) & (f < 12800)].sum(0) + 1e-14); tr = 10 * np.log10(PR[(f >= 100) & (f < 12800)].sum(0) + 1e-14)
        a = (tl - uniform_filter1d(tl, 300, mode='nearest'))[act]; b = (tr - uniform_filter1d(tr, 300, mode='nearest'))[act]
        o['pn_envcorr'] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else 1.0
    # ---------------------------------------------------------------- sample values
    o.update(sv_topk=0.0, sv_hold=0.0, sv_flat=0.0, sv_top=0.0)
    for ch in (L, R):
        x = ch.astype(np.float32); nz = np.abs(x) > 1e-6 * pk
        if nz.sum() < 1000: continue
        u, cnt = np.unique(x[nz], return_counts=True); o['sv_topk'] = max(o['sv_topk'], float(np.sort(cnt)[-1024:].sum() / nz.sum()))
        same = (x[1:] == x[:-1]) & nz[1:]; o['sv_hold'] = max(o['sv_hold'], float(same.sum() / nz.sum()))
        p1 = np.abs(x).max(); hot = np.abs(x) > 0.9 * p1; o['sv_flat'] = max(o['sv_flat'], float((hot[1:] & (np.abs(np.diff(x)) < 1e-4 * p1)).sum() / len(x))); o['sv_top'] = max(o['sv_top'], float((np.abs(x) > 0.8 * p1).mean()))
    # ---------------------------------------------------------------- a carrier: a steady line in the envelope at audio rate
    o.update(cr_stab=0.0, cr_z=0.0, cr_f=0.0)
    seg = 8192; hop = 4096; e2 = M ** 2; fr = np.fft.rfftfreq(seg, 1 / SR); selc = (fr >= 40) & (fr <= 4500); wn = np.hanning(seg); fs_, zs = [], []
    for s0 in range(0, n - seg, hop):
        v = e2[s0:s0 + seg]
        if v.mean() < 1e-4 * e2.max(): continue
        P = np.abs(np.fft.rfft((v - v.mean()) * wn)) ** 2
        if v.max() == v.min(): P[:] = 0                 # a stretch of one value (a square wave, one bit) has no line: none is made up from the rounding of the mean
        k = int(np.argmax(np.where(selc, P, 0))); fs_.append(fr[k]); zs.append(10 * np.log10(P[k] / (np.median(P[selc]) + 1e-30) + 1e-30))
    if len(fs_) >= 6:
        fs_ = np.array(fs_); fm = np.median(fs_); o['cr_stab'] = float((np.abs(fs_ - fm) <= max(1.5 * SR / seg, 0.015 * fm)).mean()); o['cr_z'] = float(np.median(zs)); o['cr_f'] = float(np.log2(fm + 1))
    # ---------------------------------------------------------------- stretches that repeat exactly (stutter)
    o.update(rp_frac=0.0, rp_contrast=0.0, rp_lag=0.0)
    m = min(n, 12 * SR); c = np.abs(ceps(M[:m])); lo, hi = int(0.02 * SR), min(int(0.75 * SR), len(c) // 2 - 1)
    if hi > lo + 100:
        segc = c[lo:hi]; med = np.median(segc); mad = np.median(np.abs(segc - med)) + 1e-12; best = (0.0, 0.0, 0)
        def rep(Lg):
            w = 480; k = (m - Lg) // w
            if k < 20 or Lg < 1: return 0.0
            a = M[:k * w].reshape(k, w); b = M[Lg:Lg + k * w].reshape(k, w); ea = (a ** 2).sum(1); eb = (b ** 2).sum(1); live = (ea > 1e-4 * ea.max()) & (eb > 1e-4 * ea.max())
            if live.sum() < 10: return 0.0
            r = (a * b).sum(1)[live] / np.sqrt(ea[live] * eb[live]); return float((r > 0.98).mean())
        for k in np.argsort(segc)[::-1][:3]:
            if (segc[k] - med) / mad < 15: break
            Lg = int(k + lo); fr_ = rep(Lg); con = fr_ - max(rep(int(round(0.618 * Lg))), rep(int(round(1.382 * Lg))))
            if con > best[1] or best[2] == 0: best = (fr_, con, Lg) if con >= best[1] else best
        o.update(rp_frac=best[0], rp_contrast=best[1], rp_lag=best[2] / SR * 1e3)
    # ---------------------------------------------------------------- the long-term spectrum: an edge (a fixed filter), the top end (exciter)
    o.update(ft_lp_slope=0.0, ft_lp_f=0.0, ft_hp_slope=0.0, ft_hp_f=0.0, ft_peak=0.0, ft_hf_db=-80.0, ft_lf_db=-80.0, ft_cent=0.0, ft_top_p95=0.0, ft_top_spread=0.0, hf_corr=0.0, hf_slope=0.0)
    if T >= 100 and act.sum() > 50:
        fe6 = 50.0 * 2 ** (np.arange(53) / 6.0); W6 = np.array([((f >= fe6[i]) & (f < fe6[i + 1])) / max(1, ((f >= fe6[i]) & (f < fe6[i + 1])).sum()) for i in range(52)])
        Pa = PM[:, act]; mp = Pa.mean(1); fw, Pw = ss.welch(M, SR, nperseg=8192, noverlap=4096)                                                    # the long-term spectrum with 6 Hz lines, so that the low sixth-octaves are filled
        lt = 10 * np.log10(np.array([Pw[(fw >= fe6[i]) & (fw < fe6[i + 1])].mean() for i in range(52)]) + 1e-20); lt = np.maximum(lt, lt.max() - 80); d = lt[6:] - lt[:-6]; fc6 = np.log2(fe6[3:49])
        ok = lt[:-6] > lt.max() - 30; k = int(np.argmin(np.where(ok, d, 0))); o['ft_lp_slope'] = float(-d[k]); o['ft_lp_f'] = float(fc6[k])              # the steepest fall over an octave that starts in a live band
        ok = lt[6:] > lt.max() - 30; k = int(np.argmax(np.where(ok, d, 0))); o['ft_hp_slope'] = float(d[k]); o['ft_hp_f'] = float(fc6[k])                 # the steepest rise that ends in a live band
        pr = lt[3:-3] - 0.5 * (lt[:-6] + lt[6:]); o['ft_peak'] = float(np.where(lt[3:-3] > lt.max() - 30, pr, 0).max())                                 # a peak standing out of its half octave either side
        tp = mp.sum() + 1e-30; o['ft_hf_db'] = float(max(-80, 10 * np.log10(mp[f >= 6000].sum() / tp + 1e-30))); o['ft_lf_db'] = float(max(-80, 10 * np.log10(mp[f < 150].sum() / tp + 1e-30)))
        o['ft_cent'] = float(np.log2((mp * f).sum() / tp + 1.0))
        Lf = 10 * np.log10(W6[18:] @ Pa + 1e-16); live6 = Lf > Lf.max(0, keepdims=True) - 30; topi = 51 - np.argmax(live6[::-1], 0); tf_ = np.log2(fe6[topi + 1])  # each frame's highest band (from 400 Hz up) within 30 dB of its strongest
        o['ft_top_p95'] = float(np.percentile(tf_, 95)); o['ft_top_spread'] = float(np.percentile(tf_, 95) - np.percentile(tf_, 50))
        mid = 10 * np.log10(Pa[(f >= 500) & (f < 3000)].sum(0) + 1e-14); hf = 10 * np.log10(Pa[(f >= 5000) & (f < 12000)].sum(0) + 1e-14)
        if mid.std() > 1e-6 and hf.std() > 1e-6: o['hf_corr'] = float(np.corrcoef(mid, hf)[0, 1]); o['hf_slope'] = float(np.polyfit(mid, hf, 1)[0])
    # ---------------------------------------------------------------- a ceiling: do the loud stretches have flatter waves than the quiet ones (saturation, distortion)
    o.update(sat_ceil=0.0, sat_crest_top=0.0, sat_crest_diff=0.0, sat_lvl_spread=0.0, sat_pk_spread=0.0)
    x = L if sl >= sr else R; nf = (len(x) - 1024) // 512 + 1
    if nf >= 50:
        fr2 = np.lib.stride_tricks.sliding_window_view(x, 1024)[::512]; rm = np.sqrt((fr2 ** 2).mean(1)) + 1e-12; pkf = np.abs(fr2).max(1) + 1e-12; a3 = rm > rm.max() * 10 ** (-30 / 20)
        if a3.sum() >= 30:
            cr = 20 * np.log10(pkf[a3] / rm[a3]); lv = 20 * np.log10(rm[a3]); q = np.argsort(lv); k3 = max(5, len(q) // 3); pd_ = 20 * np.log10(pkf[a3])
            o.update(sat_ceil=float((pkf[a3] > 0.891 * pkf.max()).mean()), sat_crest_top=float(np.median(cr[q[-k3:]])), sat_crest_diff=float(np.median(cr[q[-k3:]]) - np.median(cr[q[:k3]])),
                     sat_lvl_spread=float(np.percentile(lv, 95) - np.percentile(lv, 50)), sat_pk_spread=float(np.percentile(pd_, 95) - np.percentile(pd_, 50)))
    if not NAMES: NAMES.extend(o.keys())
    return { k: float(np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)) for k, v in o.items() }
