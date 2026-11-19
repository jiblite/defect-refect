# Test sounds with a known pitch movement, and what the reader of pitch movement (flow.py) makes of them.
#   python3 test_flow.py [names ...]                       the prototype's reader
#   python3 test_flow.py --app path/to/drmove [names ...]  the app's reader next to it (Tests/move.cpp built as drmove),
#                                                          and how far the two curves lie apart (cents, at most)
# Per sound: held = share of the moments at which the pitch is followed; err = error in cents (rms) inside the stretches
# the truth speaks of, each taken around its own average; gain = how much of the truth's movement was read (1 = all);
# apart = how far the averages of those stretches lie apart in what was read.
import os, sys, time, subprocess, tempfile, multiprocessing as mp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp
SR = 48000
def tone(dur, f0=110.0, parts=40, tilt=1.0, seed=1, jitter=0.0):
    n = int(dur * SR); t = np.arange(n) / SR; rng = np.random.default_rng(seed); x = np.zeros(n)
    for k in range(1, parts + 1):
        if k * f0 > 20000: break
        x += np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28)) * (1 + jitter * rng.standard_normal()) / k ** tilt
    return 0.2 * x / np.abs(x).max()
def noise(dur, seed=2, pink=False):
    n = int(dur * SR); rng = np.random.default_rng(seed); x = rng.standard_normal(n)
    if pink:
        X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR); X[1:] /= np.sqrt(f[1:]); X[0] = 0; x = np.fft.irfft(X, n)
    return 0.1 * x / np.abs(x).std() / 3
def fade(x, ms=10):
    k = int(ms * SR / 1000); w = np.ones(len(x)); w[:k] = np.linspace(0, 1, k); w[-k:] = np.linspace(1, 0, k); return x * w
def doppler(n, v=30.0, d=5.0):
    t = np.arange(n) / SR - n / SR / 2; vr = v * v * t / np.sqrt(d * d + (v * t) ** 2); return 1200 * np.log2(1 / (1 + vr / 343.0))
def cases():
    out = {}; n = 6 * SR; t = np.arange(n) / SR; base = tone(8.0)
    def bent(src, c): return warp.bend(src, c)[0][:len(c)]
    out['steady'] = (base[:n], np.zeros(n))
    c = 30 * np.sin(2 * np.pi * 5.5 * t); out['vibrato'] = (bent(base, c), c)
    c = np.linspace(-300, 300, n); out['glide'] = (bent(base, c), c)
    c = doppler(n); out['doppler'] = (bent(base, c), c)
    c = 600 * np.tanh((t - 3) * 1.2); out['rev'] = (bent(tone(12.0), c), c)
    # an engine: partials 22 Hz apart with some noise, revving fast
    eng = tone(14.0, f0=22.0, parts=180, tilt=0.7, seed=5) + 0.02 * noise(14.0, seed=6, pink=True)
    c = np.concatenate([np.linspace(-600, 600, 2 * SR), np.linspace(600, -200, SR), np.linspace(-200, 500, 3 * SR)]); out['engine'] = (bent(eng, c), c)
    c = np.zeros(n); out['engine steady'] = (eng[:n], c)
    # the same engine in a lot of noise
    c = doppler(n, 40.0, 8.0); out['engine in noise'] = (bent(eng, c) + 0.5 * noise(6.0, seed=8, pink=True), c)
    out['white noise'] = (noise(6.0), None); out['pink noise'] = (noise(6.0, pink=True), None)
    # falling zaps of 0.3 s, 600 cents each: apart, back to back, and each starting half way through the one before
    z = fade(bent(tone(1.0, f0=660.0, parts=12), np.linspace(300, -300, int(0.3 * SR))))
    for name, step in (('zaps apart', 0.5), ('zaps back to back', 0.3), ('zaps overlapping', 0.15)):
        x = np.zeros(n); c = np.full(n, np.nan); k = 0
        while int(k * step * SR) + len(z) <= n:
            a = int(k * step * SR); x[a:a + len(z)] += z
            if step >= 0.3: c[a:a + len(z)] = np.linspace(300, -300, len(z))
            k += 1
        out[name] = (x, c if step >= 0.3 else None)
    # two long notes a fifth apart handing over, each with its own slow bend of 100 cents
    a = bent(tone(4.0, f0=220.0, parts=20), np.linspace(-50, 50, 3 * SR)); b = bent(tone(4.0, f0=330.0, parts=20, seed=3), np.linspace(-50, 50, 3 * SR))
    x = np.zeros(n); x[:3 * SR] += fade(a, 60); x[int(2.9 * SR):int(5.9 * SR)] += fade(b, 60); c = np.full(n, np.nan); c[:int(2.9 * SR)] = np.linspace(-50, 50, 3 * SR)[:int(2.9 * SR)]; c[3 * SR:int(5.9 * SR)] = np.linspace(-50, 50, 3 * SR)[int(0.1 * SR):]
    out['two notes'] = (x, c)
    # a gliding tone with clicks and noise bursts over it
    c = np.linspace(-200, 200, n); x = bent(base, c); rng = np.random.default_rng(4)
    for k in range(8):
        a = int(rng.uniform(0.3, 5.5) * SR); x[a:a + int(0.04 * SR)] += fade(0.3 * rng.standard_normal(int(0.04 * SR)), 3)
    out['glide with bursts'] = (x, c)
    # a tune: steady notes of 0.4 s at different pitches (nothing to read: every note is its own event)
    x = np.zeros(n); k = 0
    for st in (0, 4, 7, 12, 7, 4, 0, -5, 0, 4, 7, 12, 7, 4):
        seg = fade(tone(0.4, f0=220.0 * 2 ** (st / 12.0), parts=15, seed=k), 15); a = int(k * 0.4 * SR); x[a:a + len(seg)] += seg; k += 1
    out['tune'] = (x, np.zeros(n))
    # a rev with bursts of noise over it: under a burst nothing proves how the sound moved, so the curve may stand still
    # there (it then reads too little), but it must not move any other way
    c = 600 * np.tanh((t - 3) * 1.2); x = bent(tone(12.0), c); rng = np.random.default_rng(9)
    for at in (1.2, 2.4, 2.8, 3.1, 3.5, 4.6):
        a = int(at * SR); k = int(0.12 * SR); x[a:a + k] += fade(0.15 * rng.standard_normal(k), 3)
    out['rev with bursts'] = (x, c)
    # an engine that rises slowly and drops 500 cents at once in the middle (a gear change): the drop is no movement of
    # lines that can be followed, the curve stands still across it (the two halves then lie 500 cents apart from the truth)
    c = np.where(t < 3.0, 40.0 * t, -500.0 + 40.0 * t); x = bent(eng, c); c = c.copy(); c[int(2.9 * SR):int(3.1 * SR)] = np.nan
    out['engine with a jump'] = (x, c)
    # two steady tones 3 per cent apart: their lines beat, nothing moves
    out['two tones beating'] = (0.5 * (base[:n] + tone(8.0, f0=113.3, seed=7)[:n]), np.zeros(n))
    return out
def score(p, held, c):
    """p, held on the 10 ms grid; c per sample (nan where nothing is said). Error (cents rms) inside the stretches the truth
    speaks of, each taken around its own average (err), how far those averages lie apart in what was read (off), and how much
    of the truth's movement inside the stretches was read (gain)."""
    T = len(p); ct = np.full(T, np.nan); idx = np.arange(T) * 480; ok = idx < len(c); ct[ok] = c[idx[ok]]
    say = ~np.isnan(ct); seg = np.cumsum(np.concatenate([[0], np.diff(say.astype(int)) != 0])); d2 = []; tt = []; pp = []; offs = []
    for g in np.unique(seg[say]):
        m = (seg == g) & say
        if m.sum() < 5: continue
        d = p[m] - ct[m]; offs.append(d.mean()); d2.append(d - d.mean()); tt.append(ct[m] - ct[m].mean()); pp.append(p[m] - p[m].mean())
    if not d2: return dict(held=float((held > 0).mean()), err=float('nan'), gain=float('nan'), off=float('nan'))
    d2 = np.concatenate(d2); tt = np.concatenate(tt); pp = np.concatenate(pp)
    return dict(held=float((held > 0).mean()), err=float(np.sqrt((d2 ** 2).mean())), gain=float(pp.dot(tt) / max(tt.dot(tt), 1e-9)) if tt.std() > 1 else float('nan'), off=float(np.std(offs)))
def line(name, p, held, c, dt):
    if c is None: return '%-20s held %.2f | read: rms %6.1f range %7.1f .. %7.1f | %.1f s' % (name, (held > 0).mean(), p.std(), p.min(), p.max(), dt)
    s = score(p, held, c); return '%-20s held %.2f | err %6.1f gain %5.2f apart %5.1f | read: rms %6.1f range %7.1f .. %7.1f | truth range %.0f | %.1f s' % (name, s['held'], s['err'], s['gain'], s['off'], p.std(), p.min(), p.max(), np.nanmax(c) - np.nanmin(c), dt)
def by_app(exe, x):
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        sf.write(os.path.join(d, 'x.wav'), np.stack([x, x], 1), SR, subtype='FLOAT'); t0 = time.time()
        r = subprocess.run([exe, os.path.join(d, 'x.wav'), '-', os.path.join(d, 'o'), '1', '1', '1', '1', '1'], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(r.stdout + r.stderr)
        v = np.fromfile(os.path.join(d, 'o.own.pitch.f64')); return v[:len(v) // 2], v[len(v) // 2:], time.time() - t0
def one(job):
    name, exe = job; import flow
    x, c = cases()[name]
    if exe: x = x.astype(np.float32).astype(np.float64)               # the app reads the sound from a 32-bit file: the prototype is asked about the same samples
    t0 = time.time(); p, held, e = flow.track(x)[:3]; out = [line(name, p, held, c, time.time() - t0)]
    if exe:
        q, hq, dt = by_app(exe, x); out.append(line('   the app', q, hq, c, dt) + ' | apart from the prototype: %.2f cents at most, followed or not differs on %.3f' % (np.abs(q - p).max(), ((hq > 0) != (held > 0)).mean()))
    return '\n'.join(out)
if __name__ == '__main__':
    args = sys.argv[1:]; exe = None
    if args[:1] == ['--app']: exe = args[1]; args = args[2:]
    names = [n for n in cases() if not args or any(o in n for o in args)]
    with mp.Pool(min(4, len(names))) as pool:
        for r in pool.imap(one, [(n, exe) for n in names]): print(r, flush=True)
