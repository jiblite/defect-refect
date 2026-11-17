# Data for the rule "is this an echo": clean 6 s clips as they are (no echo) and with a random echo put on, measured by echo2.measure.
#   python3 mkdata2.py part of        -> d2_<part>.jsonl
import sys, os, json, glob, zlib, numpy as np, soundfile as sf, scipy.signal as ss
sys.path.insert(0, '.'); import echo2 as echo
part, of = int(sys.argv[1]), int(sys.argv[2]); fs = sorted(glob.glob('/tmp/claude-0/v2/train/dry/*.flac'))[part::of]; fn = 'd2_%d.jsonl' % part
done = {json.loads(l)['name'] for l in open(fn)} if os.path.exists(fn) else set(); out = open(fn, 'a')
def put_echo(x, rng):
    """One channel. A single repeat or a repeat that feeds back, with the tone of the repeats changed in half the cases
    (highs or lows lost on every pass), and in one case in seven the repeat flipped."""
    D = int(round(np.exp(rng.uniform(np.log(0.035), np.log(1.5))) * 48000)); g = rng.uniform(0.15, 0.8) * (-1.0 if rng.random() < 0.15 else 1.0)
    f = rng.uniform(0.1, 0.7) if rng.random() < 0.5 else 0.0; kind = rng.choice(['flat', 'flat', 'lp', 'lp', 'hp', 'bp'])
    fc = float(np.exp(rng.uniform(np.log(1500.0), np.log(10000.0)))); fh = float(np.exp(rng.uniform(np.log(80.0), np.log(600.0))))
    def tone(v):
        if kind in ('lp', 'bp'): b, a = ss.butter(1, fc / 24000.0); v = ss.lfilter(b, a, v)
        if kind in ('hp', 'bp'): b, a = ss.butter(1, fh / 24000.0, 'high'); v = ss.lfilter(b, a, v)
        return v
    y = x.copy(); tap = x.copy(); k = 1
    while k * D < len(x) and abs(g) * f ** (k - 1) > 0.003:
        tap = tone(tap); y[k * D:] += g * f ** (k - 1) * tap[:len(x) - k * D]; k += 1
        if f == 0.0: break
    return y, dict(ms=D / 48.0, g=float(g), fb=float(f), kind=str(kind), fc=fc, fh=fh)
for i, fpath in enumerate(fs):
    name = os.path.basename(fpath)[:-5]
    if name in done: continue
    x, sr = sf.read(fpath, dtype='float64', always_2d=True); x1 = x[:, 0].copy()
    if not np.abs(x1).max() > 1e-4: continue
    rng = np.random.default_rng(zlib.crc32(name.encode())); y1, par = put_echo(x1, rng)
    c0, r0 = echo.measure(x1); c1, r1 = echo.measure(y1); r0 = [r for r, _ in r0]; r1 = [r for r, _ in r1]
    out.write(json.dumps(dict(name=name, clean=r0, wet=r1, par=par, cand0=c0[:4], cand1=c1[:4])) + '\n'); out.flush()
