# Data for the rule "is a level movement there": clean 6 s clips as they are, and with one of my own synthetic movements put on;
# everything find() measures about its candidates is written down.     python3 mkdata.py part of [corpus]   -> data/<set>_<part>.jsonl
import sys, os, json, glob, zlib, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level, synth
part, of = int(sys.argv[1]), int(sys.argv[2]); which = sys.argv[3] if len(sys.argv) > 3 else 'train'; os.makedirs('data', exist_ok=True)
fs = sorted(glob.glob('/tmp/claude-0/v2/train/dry/*.flac' if which == 'train' else '/tmp/claude-0/v2/corpus/dry/*.wav'))[part::of]; path = 'data/%s_%d.jsonl' % (which, part)
done = {json.loads(l)['name'] for l in open(path)} if os.path.exists(path) else set(); out = open(path, 'a')
def strip(m): return {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in m.items() if k not in ('g', 'prof', 'lane')} if m else None
def measure(x):
    E = level.levels(x); pan = level.find_pan(E); tog = level.candidates_together(E); return dict(pan=strip(pan), tog=[strip(m) for m in tog])
for i, f in enumerate(fs):
    name = os.path.splitext(os.path.basename(f))[0]
    if name in done: continue
    x, sr = sf.read(f, dtype='float64', always_2d=True); x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
    if not np.abs(x).max() > 1e-4: continue
    rng = np.random.default_rng(zlib.crc32(name.encode())); kind = ('trem', 'gate', 'pan')[int(rng.integers(0, 3))]; g, par = synth.make(len(x), rng, kind)
    out.write(json.dumps(dict(name=name, clean=measure(x), wet=measure(x * g), par=par)) + '\n'); out.flush()
