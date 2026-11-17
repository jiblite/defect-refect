# Data for the rule "is this an echo", round 4: measured by the app's own engine (drecho measure) on
#   clip    1057 clean 6 s training clips, as they are and with a synthetic echo
#   dry     the 198 whole corpus sounds, as they are and with three different synthetic echoes
#   <fam>   the wet sounds of every scoreboard family but delay, as they are (no echo; a reverb's discrete reflections and a comb
#           filter of 30 ms or more are echoes of their own kind and are left out of the fit later)
# One line per measured sound: kind, name, src (the sound it is made from, for splitting), var, par (the echo put on, per channel), m.
#   python3 mkdata4.py part of   -> d4_<part>.jsonl
import sys, os, json, glob, zlib, subprocess, numpy as np, soundfile as sf, scipy.signal as ss
part, of = int(sys.argv[1]), int(sys.argv[2]); fn = 'd4_%d.jsonl' % part; tool = '/tmp/claude-0/drbuild/drecho'; tmp = '/dev/shm/ep/d4_%d.wav' % part
done = {(r['kind'], r['name'], r['var']) for r in map(json.loads, open(fn))} if os.path.exists(fn) else set(); out = open(fn, 'a')
def draw(rng, dmax):
    """A single repeat or a repeat that feeds back, with the tone of the repeats changed in half the cases (highs or lows lost on
    every pass), in one case in seven the repeat flipped; in three cases in ten the right side has another echo time."""
    D = int(round(np.exp(rng.uniform(np.log(0.035), np.log(dmax))) * 48000)); g = rng.uniform(0.15, 0.8) * (-1.0 if rng.random() < 0.15 else 1.0)
    f = rng.uniform(0.1, 0.7) if rng.random() < 0.5 else 0.0; kind = str(rng.choice(['flat', 'flat', 'lp', 'lp', 'hp', 'bp']))
    fc = float(np.exp(rng.uniform(np.log(1500.0), np.log(10000.0)))); fh = float(np.exp(rng.uniform(np.log(80.0), np.log(600.0))))
    D2 = D if rng.random() < 0.7 else int(np.clip(round(D * float(rng.choice([0.5, 0.667, 0.75, 1.333, 1.5, 2.0]))), 0.035 * 48000, dmax * 48000))
    return dict(D=[D, D2], g=float(g), fb=float(f), kind=kind, fc=fc, fh=fh)
def put(x, D, p):
    def tone(v):
        if p['kind'] in ('lp', 'bp'): b, a = ss.butter(1, p['fc'] / 24000.0); v = ss.lfilter(b, a, v)
        if p['kind'] in ('hp', 'bp'): b, a = ss.butter(1, p['fh'] / 24000.0, 'high'); v = ss.lfilter(b, a, v)
        return v
    y = x.copy(); tap = x.copy(); k = 1; g, f = p['g'], p['fb']
    while k * D < len(x) and abs(g) * f ** (k - 1) > 0.003:
        tap = tone(tap); y[k * D:] += g * f ** (k - 1) * tap[:len(x) - k * D]; k += 1
        if f == 0.0: break
    return y
jobs = []; S = '/tmp/claude-0/v2/bench/sets/'
for f in sorted(glob.glob('/tmp/claude-0/v2/train/dry/*.flac')):
    n = os.path.basename(f)[:-5]; jobs += [('clip', n, n, 'clean', f), ('clip', n, n, 'echo0', f)]
for f in sorted(glob.glob('/tmp/claude-0/v2/corpus/dry/*.wav')):
    n = os.path.basename(f)[:-4]; jobs += [('dry', n, n, v, f) for v in ('clean', 'echo0', 'echo1', 'echo2')]
for fam in sorted(os.listdir(S)):
    if fam == 'delay': continue
    for name in sorted(os.listdir(S + fam)):
        m = json.load(open(S + fam + '/' + name + '/meta.json'))
        for w in 'AB':
            if os.path.exists(S + fam + '/' + name + '/' + w + '_wet.flac'): jobs.append((fam, name + '_' + w, m[w], 'clean', S + fam + '/' + name + '/' + w + '_wet.flac'))
for kind, name, src, var, path in jobs[part::of]:
    if (kind, name, var) in done: continue
    x, sr = sf.read(path, dtype='float64', always_2d=True)
    if sr != 48000: continue
    x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1); par = None
    if not np.abs(x).max() > 1e-4: continue
    if var != 'clean':
        rng = np.random.default_rng(zlib.crc32((name + var).encode())); par = draw(rng, 1.5 if kind == 'clip' else 2.2)
        x = np.stack([put(x[:, 0], par['D'][0], par), put(x[:, 1], par['D'][1], par)], 1); pk = np.abs(x).max()
        if pk > 0.99: x *= 0.99 / pk
    sf.write(tmp, x.astype(np.float32), 48000, subtype='FLOAT'); p = subprocess.run([tool, 'measure', tmp], capture_output=True, text=True)
    if p.returncode != 0: continue
    out.write(json.dumps(dict(kind=kind, name=name, src=src, var=var, par=par, m=json.loads(p.stdout))) + '\n'); out.flush()
os.path.exists(tmp) and os.remove(tmp)
