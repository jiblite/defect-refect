# Second look for the rule: candidates that pass a coarse first filter (and every candidate with the true period of a synthetic
# movement) are measured again with the full, current set of measures.     python3 mkstage2.py   -> data/stage2.jsonl
import sys, os, json, glob, zlib, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level, synth
FS = 1000.0
def match(P, rate, tol=0.012):
    r = (FS / P) / rate
    if abs(r - 1) < tol: return 1
    for m in (2, 3, 4):
        if abs(r * m - 1) < tol: return m
    for a in (1, 2, 3, 4):
        for b in (1, 2, 3, 4, 5, 6):
            if abs(r * b / a - 1) < 0.02: return -1
    return 0
rows = [json.loads(l) for f in sorted(glob.glob('data/train_*.jsonl')) for l in open(f)]; path = 'data/stage2.jsonl'
done = {json.loads(l)['name'] for l in open(path)} if os.path.exists(path) else set(); out = open(path, 'a')
def strip(m): return {k: (v if not isinstance(v, (np.floating, np.integer, np.bool_)) else float(v)) for k, v in m.items() if k not in ('g', 'prof', 'lane')}
def again(E, ms, keep):
    res = []
    for m0 in ms:
        if not keep(m0): continue
        m = level.measure_together(E, m0['P'])
        if m is None: continue
        m['share'] = m0['share']; m['excess'] = m0['excess']; res.append(strip(m))
    return res
for r in rows:
    name = r['name']
    if name in done: continue
    x, sr = sf.read('/tmp/claude-0/v2/train/dry/%s.flac' % name, dtype='float64', always_2d=True); x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
    rng = np.random.default_rng(zlib.crc32(name.encode())); kind = ('trem', 'gate', 'pan')[int(rng.integers(0, 3))]; g, par = synth.make(len(x), rng, kind); assert abs(par['rate'] - r['par']['rate']) < 1e-9
    o = dict(name=name, par=par, clean=[], wet=[])
    if any(level.first_filter(m) for m in r['clean']['tog']): o['clean'] = again(level.levels(x), r['clean']['tog'], level.first_filter)
    if kind in ('trem', 'gate'):
        E = None
        for m0 in r['wet']['tog']:
            mt = match(m0['P'], par['rate'])
            if mt == 0 and not level.first_filter(m0): continue
            if E is None: E = level.levels(x * g)
            m = level.measure_together(E, m0['P'])
            if m is None: continue
            m['share'] = m0['share']; m['excess'] = m0['excess']; m = strip(m); m['match'] = mt; o['wet'].append(m)
    out.write(json.dumps(o) + '\n'); out.flush()
