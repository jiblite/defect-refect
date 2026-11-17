# Data for the rule "is this an echo", an addition to round 4 (mkdata4.py): echoes in time with the sound. A delay on rhythmic
# material is nearly always set to the tempo, so its echoes fall on the grid of the sound's own repeats; a random echo time (as in
# mkdata4.py) never does. Per whole corpus sound that has a repeat of its own (the strongest lag of 150 ms or more that stands
# out on the clean sound, from d4_*.jsonl): two echoes at a simple fraction of that period, measured by the app's engine.
#   python3 mkdata5.py part of   -> d5_<part>.jsonl   (rows as in mkdata4.py, kind 'dry', var 'sync0' / 'sync1'; par also holds P and r)
import sys, os, json, glob, zlib, subprocess, numpy as np, soundfile as sf, scipy.signal as ss
part, of = int(sys.argv[1]), int(sys.argv[2]); fn = 'd5_%d.jsonl' % part; tool = '/tmp/claude-0/drbuild/drecho'; tmp = '/dev/shm/ep/d5_%d.wav' % part
done = {(r['name'], r['var']) for r in map(json.loads, open(fn))} if os.path.exists(fn) else set(); out = open(fn, 'a')
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
period = {}
for f in sorted(glob.glob('d4_[0-9]*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        if r['kind'] != 'dry' or r['var'] != 'clean': continue
        c = [k for ch in r['m']['ch'] for k in ch['cand'] if k['q'] >= 7200]
        if c: period[r['name']] = max(c, key=lambda k: k['score'])['q']
R = [0.5, 0.25, 0.75, 1.0 / 3, 2.0 / 3, 0.375, 0.125, 1.5]
jobs = [(n, v) for n in sorted(period) for v in ('sync0', 'sync1')]
for name, var in jobs[part::of]:
    if (name, var) in done: continue
    x, sr = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % name, dtype='float64', always_2d=True)
    if sr != 48000: continue
    x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1); P = period[name]
    rng = np.random.default_rng(zlib.crc32((name + var).encode())); ok = [r for r in R if 0.035 * 48000 <= P * r <= 2.2 * 48000]
    if not ok: continue
    r0 = float(rng.choice(ok)); r1 = r0 if rng.random() < 0.7 else float(rng.choice(ok)); D = [int(round(P * r0)), int(round(P * r1))]
    g = rng.uniform(0.15, 0.8) * (-1.0 if rng.random() < 0.15 else 1.0); fb = rng.uniform(0.1, 0.7) if rng.random() < 0.5 else 0.0
    par = dict(D=D, g=float(g), fb=float(fb), kind=str(rng.choice(['flat', 'flat', 'lp', 'lp', 'hp', 'bp'])), fc=float(np.exp(rng.uniform(np.log(1500.0), np.log(10000.0)))),
               fh=float(np.exp(rng.uniform(np.log(80.0), np.log(600.0)))), P=int(P), r=[r0, r1])
    y = np.stack([put(x[:, 0], D[0], par), put(x[:, 1], D[1], par)], 1); pk = np.abs(y).max()
    if pk > 0.99: y *= 0.99 / pk
    sf.write(tmp, y.astype(np.float32), 48000, subtype='FLOAT'); p = subprocess.run([tool, 'measure', tmp], capture_output=True, text=True)
    if p.returncode != 0: continue
    out.write(json.dumps(dict(kind='dry', name=name, src=name, var=var, par=par, m=json.loads(p.stdout))) + '\n'); out.flush()
os.path.exists(tmp) and os.remove(tmp)
