# Echo engine, prototype 2 (the form that is ported to C++: power-of-two transforms, one stretch of the sound, cheap sums).
# Finds echoes (30 ms to 2.5 s) on ONE sound by the ripple they leave in its spectrum, and takes them out or puts them on
# another sound (a sound plus quieter copies of itself is a minimum-phase filter, and such a filter is given by its ripple).
#   stretch(x)        the part of the sound that is looked at (all of it up to 19 s, else the 19 s with the most energy)
#   cep(x1)           the ripple picture of one channel: cepstrum, lag in samples
#   candidates(...)   lags that stand out of it, each with how loud the copy is where the sound has its level
#   measure(x1)       the most echo-like lags with everything the rule looks at
#   find(x, rule)     per channel the echo part of the picture (windows around the echo's lags), or None
#   response(E, n)    the echo filter's impulse response (the sound itself at time 0, then the repeats)
#   apply(x, E, s)    s = -1: the found echoes out of x; s = +1: onto x (as a power of the filter; the app scales the repeats instead)
import numpy as np, scipy.fft as sfft, scipy.ndimage as ndi
SR = 48000; QMIN = 1440; QMAX = 120000; HALF = 96; NMAX = 912000; MC = 4096

def pow2(n):
    p = 1
    while p < n: p *= 2
    return p

def stretch(x):
    """x: (n, 2) or (n,). Returns (start, length) of the part that is analysed."""
    n = len(x)
    if n <= NMAX: return 0, n
    e = (x ** 2).sum(1) if x.ndim == 2 else x ** 2; hop = 24000; nb = n // hop; b = e[:nb * hop].reshape(nb, hop).sum(1); k = NMAX // hop
    c = np.concatenate([[0.0], np.cumsum(b)]); s = int(np.argmax(c[k:] - c[:-k])); return s * hop, NMAX

def cep(x1):
    """Cepstrum of one channel (trend of the log spectrum taken off, dead parts of the spectrum left out).
    Returns (c, nfft, PsC): PsC = the smoothed power spectrum summed onto MC/2+1 coarse bins (where the sound has its level)."""
    n = len(x1); nfft = pow2(2 * n + 2 * QMAX)
    P = np.abs(sfft.rfft(x1, nfft)) ** 2
    w = max(3, int(round(nfft / QMIN / 2)) | 1)                 # half the widest ripple period we look at, in bins
    Ps = ndi.uniform_filter1d(P, 4 * w + 1, mode='nearest'); top = Ps.max(); r = nfft // MC
    if not top > 0: return np.zeros(nfft), nfft, np.zeros(MC // 2 + 1)
    live = Ps > top * 1e-10                                     # where the file holds anything at all (100 dB under the top)
    m = ndi.uniform_filter1d(live.astype(float), 4 * w + 1, mode='nearest')      # soft edges
    L = np.log(np.maximum(P, Ps * 1e-4)); L = (L - ndi.uniform_filter1d(L, 8 * w + 1, mode='nearest')) * m
    # coarse bin j holds the fine bins j*r - r/2 .. j*r + r/2 - 1
    pad = np.concatenate([np.zeros(r // 2), Ps[:nfft // 2], np.zeros(r // 2)]); PsC = np.concatenate([pad[:(MC // 2) * r].reshape(MC // 2, r).sum(1), [Ps[nfft // 2 - r // 2:nfft // 2 + 1].sum()]])
    return sfft.irfft(L, nfft), nfft, PsC

def background(c, q1):
    """Typical size of the cepstrum around each lag (median of |c| over about 85 ms), for telling a peak from the rest."""
    a = np.abs(c[:q1]); med = ndi.median_filter(a[::16], 257, mode='nearest')
    return np.maximum(np.interp(np.arange(q1), np.arange(len(med)) * 16, med), 1e-12) / 0.6745       # as a standard deviation

def window(c, q, bg=None, t=3.0):
    """The cepstrum around lag q, with what does not stand out of the background turned down (soft threshold at t times it)."""
    w = c[q - HALF:q + HALF + 1] * np.hanning(2 * HALF + 1) ** 0.5
    if bg is None: return w
    b = t * bg[q - HALF:q + HALF + 1]
    return w * np.maximum(0.0, 1.0 - (b / np.maximum(np.abs(w), 1e-30)) ** 2)

def ripple(w):
    """|W|^2 of a lag window on the coarse bins: the size of the ripple this window makes, over frequency (its mean square is half of this)."""
    z = np.zeros(MC); z[:HALF + 1] = w[HALF:]; z[MC - HALF:] = w[:HALF]; return np.abs(sfft.rfft(z)) ** 2

def loudness(w, PsC):
    """How loud is the copy with lag window w against the sound itself, where the sound has its level (1 = as loud)."""
    return float(np.sqrt((PsC * ripple(w)).sum() / max(PsC.sum(), 1e-30)))

def candidates(c, nfft, PsC, thr=8.0, most=8):
    """Lags from QMIN up where the cepstrum stands out: dicts (q, score, a = loudness of the copy, h = value at the peak), by score."""
    q1 = min(QMAX + HALF, nfft // 2 - 2 * HALF); bg = background(c, q1 + HALF); a = np.abs(c[:q1])
    e = np.sqrt(ndi.uniform_filter1d(a ** 2, 9, mode='nearest') * 9)             # size over +-4 samples
    sc = e / bg[:q1]; sc[:QMIN] = 0; sc[min(QMAX, nfft // 2 - 2 * HALF) - 2 * HALF - 3:] = 0; out = []     # a base lag must leave room for its window in series()
    for _ in range(most):
        k = int(np.argmax(sc))
        if sc[k] < thr: break
        j = k - 4 + int(np.argmax(a[k - 4:k + 5])); out.append(dict(q=j, score=float(sc[k]), h=float(c[j]), a=loudness(window(c, j, bg), PsC))); sc[max(0, k - 2 * HALF):k + 2 * HALF] = 0
    return out, bg

def rahmonic(c, bg, q, lo=96, kmax=8, part=0.5):
    """Is the peak at lag q one of a series that starts at a shorter lag (a steady pitch gives peaks at its period and every
    multiple)? True when the cepstrum also stands out at q/k for some k = 2..kmax, at least part as far as at q."""
    here = np.abs(c[q - 2:q + 3]).max() / bg[q]
    for k in range(2, kmax + 1):
        j = int(round(q / k))
        if j < lo: break
        r = max(2, int(round(0.004 * j)))
        if np.abs(c[j - r:j + r + 1]).max() / bg[j] > part * here: return True
    return False

def only_removes(x1, y1, hop=2400):
    """Taking a true echo out only ever takes level away. Returns (added, removed): energy that appears in y where x had
    less, and energy that went, over 50 ms steps, as shares of x's energy."""
    n = min(len(x1), len(y1)) // hop * hop; ea = (x1[:n] ** 2).reshape(-1, hop).mean(1); eb = (y1[:n] ** 2).reshape(-1, hop).mean(1); tot = max(ea.sum(), 1e-30)
    return float(np.maximum(eb - ea, 0).sum() / tot), float(np.maximum(ea - eb, 0).sum() / tot)

def series(c, nfft, bg, D, kmax=8):
    """The cepstrum windows of an echo with base lag D: the lag itself and its multiples (those that stand out)."""
    W = {}; pk = []
    for k in range(1, kmax + 1):
        q = D * k
        if q + 2 * HALF > min(QMAX, nfft // 2 - 2 * HALF): break
        j = q - 6 + int(np.argmax(np.abs(c[q - 6:q + 7]))) if k > 1 else q
        w = window(c, j, bg)
        if k > 1 and not np.any(w): continue
        W[j] = w; pk.append((j, float(c[j]), k))
    return W, pk

def evenness(w, PsC, bands=24):
    """How evenly is the copy there across the spectrum (a true echo copies everything)? The ripple's size in bands of
    equal width on a log scale from 100 Hz to 16 kHz, over the bands where the sound has level: returns (share of those
    bands where the copy is at least half its median size, their number)."""
    R = ripple(w); f = np.arange(MC // 2 + 1) * SR / MC; ed = np.geomspace(100.0, 16000.0, bands + 1); lev = []; siz = []
    for lo, hi in zip(ed[:-1], ed[1:]):
        m = (f >= lo) & (f < hi); lev.append(PsC[m].mean()); siz.append(np.sqrt(R[m].mean()))
    lev = np.array(lev); siz = np.array(siz); live = lev > lev.max() * 1e-4
    if live.sum() < 3: return 0.0, int(live.sum())
    return float((siz[live] >= 0.5 * np.median(siz[live])).mean()), int(live.sum())

def pairs(x1, D, cmin=0.9, ctl=0.7):
    """Every moment of the sound against the moment D later, in bands of 8 bins up to 16 kHz: where the later one is a copy
    of the earlier one (coherence over cmin), how much quieter or louder is it? An echo's copies are all quieter by the same
    amount; notes that are played again are about as loud as before. Only moments that can testify count: a held note is
    "a copy of itself" at every lag, so cells that are also coherent half a frame before or after the lag are left out.
    Returns dict: info = part of the sound's energy that can testify, share = part of that with a copy at the lag, med =
    typical level of the copies (dB against the original), iqr = how much that varies (dB), up = part of the copies that
    are as loud as the original or louder (within 1 dB), n = number of copies, eff = how many of them carry the weight."""
    n = 2048 if D >= 2730 else 1024; hop = n // 4; h2 = n // 2; m = len(x1) - D - n - h2
    z = dict(share=0.0, med=0.0, iqr=0.0, up=0.0, n=0, info=0.0, eff=0.0)
    if m < h2 + 4 * hop: return z
    idx = np.arange(h2, m, hop)[:, None] + np.arange(n)[None, :]; w = np.hanning(n)
    X = sfft.rfft(x1[idx] * w, axis=1); Y = sfft.rfft(x1[idx + D] * w, axis=1); Y1 = sfft.rfft(x1[idx + D + h2] * w, axis=1); Y2 = sfft.rfft(x1[idx + D - h2] * w, axis=1)
    nb = (int(16000 * n / SR) - 4) // 8; sl = slice(4, 4 + nb * 8)
    def band(a): return a[:, sl].reshape(len(a), nb, 8).sum(2)
    EX = band(np.abs(X) ** 2); EY = band(np.abs(Y) ** 2); E1 = band(np.abs(Y1) ** 2); E2 = band(np.abs(Y2) ** 2)
    def coh(Yv, Ev): return np.abs(band(Yv * np.conj(X))) / np.sqrt(np.maximum(EX * Ev, 1e-60))
    c0 = coh(Y, EY); c1 = coh(Y1, E1); c2 = coh(Y2, E2); top = EX.max()
    live = EX > top * 1e-5; inf = live & ~(np.maximum(c1, c2) > ctl); tot = EX[live].sum(); ti = EX[inf].sum(); ok = inf & (c0 > cmin) & (EY > top * 1e-7)
    z['info'] = float(ti / tot) if tot > 0 else 0.0
    if not ti > 0: return z
    z['share'] = float(EX[ok].sum() / ti); z['n'] = int(ok.sum())
    if ok.sum() < 8: return z
    r = 10 * np.log10(EY[ok] / EX[ok]); wt = EX[ok]; o = np.argsort(r, kind='stable'); cw = np.cumsum(wt[o]) / wt.sum(); q = lambda v: float(r[o][min(np.searchsorted(cw, v), len(r) - 1)])
    z.update(med=q(0.5), iqr=q(0.75) - q(0.25), up=float(wt[r > -1.0].sum() / wt.sum()), eff=float(wt.sum() ** 2 / max((wt ** 2).sum(), 1e-300))); return z

def measure(x1, thr=8.0, amin=0.05, amax=0.95, most=3):
    """One channel (the stretch): the most echo-like lags (loudest copy first, at most `most`), each with everything
    measured about it. Returns (all candidates, list of (record, echo))."""
    c, nfft, PsC = cep(x1); cand, bg = candidates(c, nfft, PsC, thr); out = []
    for k in cand: k['rah'] = rahmonic(c, bg, k['q'])
    ok = sorted([k for k in cand if amin <= k['a'] <= amax and not k['rah']], key=lambda k: -k['a'])[:most]
    for base in ok:
        W, pk = series(c, nfft, bg, base['q']); e = dict(win=W, peaks=pk, base=base)
        a2 = loudness(W[pk[1][0]], PsC) if len(pk) > 1 and pk[1][2] == 2 else 0.0
        y1 = apply1(x1, e, -1.0)[:len(x1)]; added, removed = only_removes(x1, y1); ev, nb = evenness(W[base['q']], PsC)
        others = sum(1 for k in cand if k is not base and k['score'] > 0.5 * base['score'] and abs(k['q'] / base['q'] - round(k['q'] / base['q'])) > 0.02)
        pr = pairs(x1, base['q'])
        out.append((dict(q=base['q'], ms=base['q'] / 48.0, score=base['score'], a=base['a'], h=base['h'], a2=a2, added=added, removed=removed, even=ev, bands=nb, others=others, npk=len(pk),
                         secs=len(x1) / SR, pshare=pr['share'], pmed=pr['med'], piqr=pr['iqr'], pup=pr['up'], pn=pr['n'], pinfo=pr['info'], peff=pr['eff']), e))
    return cand, out

NAMES = ['logscore', 'a', 'a2/a', 'add/rem', 'logrem', 'others', 'loglag', 'even', 'npk', 'bands', 'pshare', 'pdiff', 'absdiff', 'piqr', 'pup', 'logpn', 'fits', 'pinfo', 'logpeff']
def feats(r):
    """What the rule looks at (nan = cannot be said: fewer than 8 copies)."""
    a = max(r['a'], 1e-6); pn = r['pn']; nan = float('nan'); pd = (r['pmed'] - 20 * np.log10(a)) if pn >= 8 else nan
    return [np.log10(r['score']), r['a'], r['a2'] / a, min(r['added'] / max(r['removed'], 1e-4), 3.0), np.log10(max(r['removed'], 1e-4)), min(r['others'], 6), np.log10(r['ms']), r['even'], min(r['npk'], 12),
            r['bands'], r['pshare'], pd, abs(pd), r['piqr'] if pn >= 8 else nan, r['pup'] if pn >= 8 else nan, np.log10(pn + 1.0), r['secs'] * 1000.0 / r['ms'], r['pinfo'], np.log10(r.get('peff', 0.0) + 1.0)]

def accept(r):
    """The first hand rule (kept for comparison): quieter than the sound, later copies quieter still, and taking it out only takes level away."""
    return r['a'] <= 0.85 and r['a2'] <= 0.8 * r['a'] and r['added'] <= 0.25 * r['removed']

def find(x, rule=accept, thr=8.0, info=None):
    """x: (n, 2). Per channel the echo part of the cepstrum or None when nothing echo-like stands out. Looks at stretch(x)."""
    s, n = stretch(x); E = []
    for ch in range(2):
        cand, recs = measure(x[s:s + n, ch], thr); got = None
        for r, e in recs:
            if rule(r): got = e; got['rec'] = r; break
        if info is not None: info.append(dict(ch=ch, cand=cand, recs=[r for r, _ in recs]))
        E.append(got)
    return E

def log_h(E, nfft, s=1.0, tscale=1.0):
    """log of the echo filter on an nfft grid from the kept cepstrum windows (minimum phase: causal part only)."""
    ch = np.zeros(nfft)
    for q, w in E['win'].items():
        q = int(round(q * tscale))
        if q + HALF + 1 <= nfft // 2: ch[q - HALF:q + HALF + 1] += s * w
    return sfft.rfft(ch, nfft)

def response(E, nfft, s=1.0, tscale=1.0):
    """The echo filter to the power s as an impulse response of nfft samples (what lies beyond wraps around)."""
    return sfft.irfft(np.exp(log_h(E, nfft, s, tscale)), nfft)

def apply1(x1, E, s=-1.0):
    n = len(x1); nfft = pow2(n + 2 * QMAX + SR)
    return sfft.irfft(sfft.rfft(x1, nfft) * np.exp(log_h(E, nfft, s)), nfft)

def apply(x, E, s=-1.0):
    """s = -1: the found echoes out of x. s = +1: onto x. Any s scales the echoes (as a power of the filter)."""
    n = len(x); y = np.zeros((pow2(n + 2 * QMAX + SR), 2))
    for ch in range(2):
        if E[ch] is None: y[:n, ch] = x[:, ch]
        else: y[:, ch] = apply1(x[:, ch], E[ch], s)
    return y
