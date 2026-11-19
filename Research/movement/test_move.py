# Scores for the Loudness, Stereo, Tone and Spectral faders: sounds that stand still, a known movement put on them,
# then (1) is the movement read, (2) does far left take it out, (3) does far right put it on a second sound.
#   python3 test_move.py [loudness|stereo|tone|spectral ...]
import sys, numpy as np
import move, tone
SR = 48000

def pink(dur, seed=1, ch=2, decor=0.3):
    """Pink noise, stereo: a shared part and a part of its own per channel (decor: how much of the power is not shared)."""
    n = int(dur * SR); rng = np.random.default_rng(seed)
    def one():
        X = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1 / SR); X[1:] /= np.sqrt(f[1:]); X[0] = 0; v = np.fft.irfft(X, n); return v / v.std()
    m = one(); return 0.1 * np.stack([np.sqrt(1 - decor) * m + np.sqrt(decor) * one() for _ in range(ch)], 1)

def chord(dur, seed=2, f0s=(110.0, 164.8, 220.0, 277.2)):
    n = int(dur * SR); t = np.arange(n) / SR; rng = np.random.default_rng(seed); out = np.zeros((n, 2))
    for f0 in f0s:
        for k in range(1, 30):
            if k * f0 > 16000: break
            a = 1.0 / k; ph = rng.uniform(0, 6.28); pan = rng.uniform(0.3, 0.7); s = a * np.sin(2 * np.pi * k * f0 * t + ph); out[:, 0] += s * np.sqrt(1 - pan); out[:, 1] += s * np.sqrt(pan)
    return 0.1 * out / np.abs(out).std()

def pulses(dur, rate, seed=3):
    """A low engine in the crudest form: a short decaying burst every 1/rate s."""
    n = int(dur * SR); rng = np.random.default_rng(seed); x = np.zeros(n); k = int(0.02 * SR); b = rng.standard_normal(k) * np.exp(-np.arange(k) / (0.004 * SR))
    for a in np.arange(0, n - k, SR / rate).astype(int): x[a:a + k] += b
    return 0.1 * np.stack([x, x], 1) / x.std()

def level_curve(y): L = move.loud(y); return L['db'] - L['ref'], L
def rms(v): return float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))

def test_loudness():
    print('LOUDNESS (dB rms; a movement of the size given is put on a steady sound)')
    n = 6 * SR; t = np.arange(n) / SR
    moves = {'swell 0.5 Hz +-6 dB': 6 * np.sin(2 * np.pi * 0.5 * t), 'pass-by +14 dB': 14 * np.exp(-0.5 * ((t - 3) / 0.7) ** 2) - 7, 'tremolo-like 6 Hz +-4 dB': 4 * np.sin(2 * np.pi * 6 * t),
             'fade out 30 dB': -30 * t / 6, 'steps 10 dB': 10 * (np.floor(t * 1.5) % 2) - 5, 'hits (decay 150 ms)': -60 * ((t * 2.5) % 1.0) / 0.9}
    srcs = {'noise': pink(6.0), 'chord': chord(6.0)}
    for name, g in moves.items():
        for sname, src in srcs.items():
            x = src * 10 ** (g / 20)[:, None]; own = move.find(x); k = np.minimum(np.arange(len(own['loud']['db'])) * move.LSTEP, n - 1); gt = g[k]
            read = own['loud']['db'] - own['loud']['ref']; base = move.loud(src); steady = base['db'] - base['ref']; ok = gt > gt.max() - 40
            d = read - steady - gt; e_read = rms((d - d[ok].mean())[ok])
            flat = move.apply(x, own, None, dict(loudness=0.0)); lf, _ = level_curve(flat); okf = gt > max(gt.max() - 24, move.LIFT_FULL + 3)
            e_flat = rms((lf - lf[okf].mean())[okf]); e_was = rms((read - read[okf].mean())[okf])
            other = srcs['chord' if sname == 'noise' else 'noise']; oo = move.find(other); put = move.apply(other, oo, own, dict(loudness=2.0)); lp, _ = level_curve(put)
            fo = move.loud(move.apply(other, oo, None, dict(loudness=0.0))); d = lp - (fo['db'] - fo['ref']) - gt; e_put = rms((d - d[ok].mean())[ok]); same = float(np.abs(move.apply(x, own, None, dict(loudness=1.0)) - x).max())
            print('  %-26s on %-5s | read off by %4.2f | flat: moves %5.2f (was %5.2f) | onto the other sound: off by %4.2f | middle changes %.0e' % (name, sname, e_read, e_flat, e_was, e_put, same))
    for rate in (13.0, 25.0, 40.0):
        x = pulses(4.0, rate); own = move.find(x); g = move.loud_gain(own['loud']['db'] - own['loud']['ref'], None, 0.0); mid = g[100:-100]
        print('  pulses at %2.0f Hz: flat moves the gain by %.2f dB rms, %.2f dB peak to peak (should leave the pulses alone)' % (rate, mid.std(), np.ptp(mid)))

def test_stereo():
    print('STEREO (dB rms)')
    n = 6 * SR; t = np.arange(n) / SR
    for sname, src in (('noise', pink(6.0, seed=4)), ('chord', chord(6.0, seed=5))):
        for name, pan, wid in (('pan sweep +-10 dB, 0.3 Hz', 10 * np.sin(2 * np.pi * 0.3 * t), np.zeros(n)), ('pass left to right 16 dB', 16 * np.tanh((t - 3) * 1.2), np.zeros(n)), ('width swell +-6 dB', np.zeros(n), 6 * np.sin(2 * np.pi * 0.4 * t)), ('pan 2 Hz +-6 dB with width', 6 * np.sin(2 * np.pi * 2 * t), 4 * np.sin(2 * np.pi * 0.25 * t))):
            m = 0.5 * (src[:, 0] + src[:, 1]); s = 0.5 * (src[:, 0] - src[:, 1]) * 10 ** (wid / 20); b = 10 ** (pan / 40); x = np.stack([(m + s) / b, (m - s) * b], 1)
            own = move.find(x); S = own['space']; k = np.minimum(np.arange(len(S['pan'])) * move.STEP, n - 1); e_pan = rms((S['pan'] - S['pan0']) - (pan[k] - pan[k].mean())); e_w = rms((S['width'] - S['width0']) - (wid[k] - wid[k].mean()))
            flat = move.apply(x, own, None, dict(stereo=0.0)); Sf = move.space(flat)
            other = chord(6.0, seed=7) if sname == 'noise' else pink(6.0, seed=8); oo = move.find(other); put = move.apply(other, oo, own, dict(stereo=2.0)); Sp = move.space(put)
            e_pp = rms((Sp['pan'] - Sp['pan0']) - (pan[k] - pan[k].mean())); e_pw = rms((Sp['width'] - Sp['width0']) - (wid[k] - wid[k].mean()))
            lv0 = move.loud(x)['db']; lv1 = move.loud(flat)['db']
            print('  %-28s on %-5s | read off by pan %4.2f width %4.2f | flat: pan moves %4.2f (was %5.2f), width %4.2f (was %4.2f), level changed by %4.2f | onto the other: pan off %4.2f width off %4.2f' % (name, sname, e_pan, e_w, Sf['pan'].std(), S['pan'].std(), Sf['width'].std(), S['width'].std(), rms(lv1 - lv0), e_pp, e_pw))
    mono = pink(6.0, seed=9, decor=0.0)[:, 0]; x = pink(6.0, seed=4); b = 10 ** (10 * np.sin(2 * np.pi * 0.3 * t) / 40); x = np.stack([x[:, 0] / b, x[:, 1] * b], 1)
    put = move.apply(mono, move.find(mono), move.find(x), dict(stereo=2.0)); Sp = move.space(put); k = np.minimum(np.arange(len(Sp['pan'])) * move.STEP, n - 1)
    print('  a mono sound takes the pan sweep: off by %.2f dB' % rms(Sp['pan'] - 10 * np.sin(2 * np.pi * 0.3 * t[k])))

def band_gains(n, kind):
    t = np.arange(n) / SR; T = 1 + n // move.STEP; tt = np.arange(T) * move.STEP / SR; b = np.arange(move.NBAND)
    if kind == 'tilt 1 Hz +-1.5 dB/oct': return 0.75 * (b[None, :] - 8.5) * np.sin(2 * np.pi * 1.0 * tt)[:, None]
    if kind == 'bright to dark over 6 s': return 1.0 * (b[None, :] - 8.5) * (1 - 2 * tt / tt[-1])[:, None] * 0.6
    if kind == 'a 12 dB bump sweeping up': return 12 * np.exp(-0.5 * ((b[None, :] - (3 + 12 * tt / tt[-1])[:, None]) / 1.5) ** 2)

def test_tone():
    print('TONE (dB rms over bands and time)')
    n = 6 * SR
    for sname, src in (('noise', pink(6.0, seed=11)), ('chord', chord(6.0, seed=12))):
        for kind in ('tilt 1 Hz +-1.5 dB/oct', 'bright to dark over 6 s', 'a 12 dB bump sweeping up'):
            g = band_gains(n, kind); x = put_bands(src, g)
            own = move.find(x); base = move.tone_move(src)['m']; wb = own['tone']['w']
            want = (g - (g @ wb / wb.sum())[:, None]); want = (want - want.mean(0, keepdims=True)) * (wb > 0.5)[None, :]
            m = own['tone']['m']; e_read = rms(((m - base) * (wb > 0.5)[None, :]) - want)
            flat = move.apply(x, own, None, dict(tone=0.0)); mf = move.tone_move(flat)['m']
            other = chord(6.0, seed=13) if sname == 'noise' else pink(6.0, seed=14); oo = move.find(other); put = move.apply(other, oo, own, dict(tone=2.0)); mp = move.tone_move(put)['m']
            lv0 = move.loud(x)['db']; lv1 = move.loud(flat)['db']
            print('  %-26s on %-5s | size %4.2f | read off by %4.2f | flat: moves %4.2f (the steady sound itself: %4.2f), level changed by %4.2f | onto the other: off by %4.2f' % (kind, sname, rms(want), e_read, rms(mf), rms(base), rms(lv1 - lv0), rms((mp - m) * ((wb > 0.5) & (oo['tone']['w'] > 0.5))[None, :])))

def put_bands(x, g):
    """Band gains (T, 18) dB put on x without keeping the moment's power (a test sound's movement)."""
    import numpy as np
    n = len(x); NT = move.NT; hop = NT // 4; w = np.sqrt(move._hann(NT)); J = 1 + (n + NT) // hop; yp = np.vstack([np.zeros((NT, 2)), x, np.zeros((NT + hop, 2))])
    Wn = move._W / np.maximum(move._W.sum(0), 1e-9)[None, :]; out = np.zeros_like(yp); tj = np.clip((np.arange(J) * hop - NT // 2) / move.STEP, 0, len(g) - 1)
    i0 = np.floor(tj).astype(int); i1 = np.minimum(i0 + 1, len(g) - 1); fr = (tj - i0)[:, None]; G = 10 ** (((g[i0] * (1 - fr) + g[i1] * fr) @ Wn) / 20.0)
    idx = np.arange(J)[:, None] * hop + np.arange(NT)[None, :]
    for c in range(2): np.add.at(out[:, c], idx, np.fft.irfft(np.fft.rfft(yp[idx, c] * w, axis=1) * G, NT, axis=1) * w)
    return out[NT:NT + n] / 2.0

def third_oct(S):
    """A long-term spectrum (dB on tone.FB) in octave bands, 63 Hz .. 8 kHz (the Spectral fader works in broad strokes)."""
    fc = 63.0 * 2 ** np.arange(8.0); P = 10 ** (S / 10); return 10 * np.log10(np.array([P[(tone.FB >= f / 2 ** 0.5) & (tone.FB < f * 2 ** 0.5)].mean() for f in fc])), fc

def test_spectral():
    print('SPECTRAL (dB rms over octave bands, 63 Hz to 8 kHz, the level taken out)')
    def eq(x, curve_fn): return tone.put_curve(x, tone.FB, curve_fn(np.maximum(tone.FB, 1.0)), 1.0)
    shapes = {'treble shelf +9 dB': lambda f: 9 / (1 + (3000.0 / f) ** 2), 'tilt -3 dB/oct': lambda f: -3 * np.log2(f / 1000.0), 'bump +10 dB at 500 Hz': lambda f: 10 * np.exp(-0.5 * (np.log2(f / 500.0) / 0.7) ** 2), 'none': lambda f: 0 * f}
    def dev(S, ref, both=None):                 # over the bands in which both sounds have something (within 35 dB of their strongest band)
        a, _ = third_oct(S); b, _ = third_oct(ref); ok = both if both is not None else (a > a.max() - 35) & (b > b.max() - 35); d = (a - b)[ok]; return rms(d - d.mean())
    for sname, src in (('noise', pink(6.0, seed=21)), ('chord', chord(6.0, seed=22))):
        for name, fn in shapes.items():
            x = eq(src, fn); own = move.find(x); flat = move.apply(x, own, None, dict(spectral=0.0)); Sf = tone.long_term_db(flat)
            other = chord(6.0, seed=23) if sname == 'noise' else pink(6.0, seed=24); oo = move.find(other); put = move.apply(other, oo, own, dict(spectral=2.0)); Sp = tone.long_term_db(put)
            lv = move.loud(flat)['ref'] - move.loud(x)['ref']; a, _ = third_oct(own['spec']); b, _ = third_oct(oo['spec']); both = (a > a.max() - 35) & (b > b.max() - 35)
            print('  %-22s on %-5s | from neutral: %5.2f, flat: %5.2f (level %+5.2f dB) | the other sound from this colour: %5.2f, far right: %5.2f (%d of 8 bands)' % (name, sname, dev(move.colour(own['spec']), move.NEUTRAL, a > a.max() - 35), dev(move.colour(Sf), move.NEUTRAL, a > a.max() - 35), lv, dev(move.colour(oo['spec']), move.colour(own['spec']), both), dev(move.colour(Sp), move.colour(own['spec']), both), both.sum()))

if __name__ == '__main__':
    which = sys.argv[1:] or ['loudness', 'stereo', 'tone', 'spectral']
    for w in which: globals()['test_' + w]()
