# Data for the pan rule: what find_pan measures on every clean 6 s train clip, and on the same clip with one of my synthetic pans put on.
#   python3 mkpan.py   -> data/pan_train.jsonl
import sys, os, json, glob, zlib, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level, synth
os.makedirs('data', exist_ok=True); path = 'data/pan_train.jsonl'
import time
while not os.path.exists('data/stage2.done'): time.sleep(2)
fs = sorted(glob.glob('/tmp/claude-0/v2/train/dry/*.flac'))[0::2]; done = {json.loads(l)['name'] for l in open(path)} if os.path.exists(path) else set(); out = open(path, 'a')
def strip(m): return {k: (v if not isinstance(v, (np.floating, np.integer, np.bool_)) else float(v)) for k, v in m.items() if k not in ('g', 'prof', 'lane')} if m else None
for f in fs:
    name = os.path.splitext(os.path.basename(f))[0]
    if name in done: continue
    x, sr = sf.read(f, dtype='float64', always_2d=True); x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
    if not np.abs(x).max() > 1e-4: continue
    rng = np.random.default_rng(zlib.crc32(name.encode()) + 1); g, par = synth.make(len(x), rng, 'pan')
    out.write(json.dumps(dict(name=name, clean=strip(level.find_pan(level.levels(x))), wet=strip(level.find_pan(level.levels(x * g))), par=par)) + '\n'); out.flush()
