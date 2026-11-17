# The decision's numbers on many sounds: (1) the clean corpus sounds as they are, (2) the same sounds with a synthetic flanger of
# random settings (made with this engine's own moving comb: sweep._run), (3) clean training clips. One line per sound in stats_<part>.jsonl.
#   python3 stats.py clean | synth | train | cleanfp [every]      (cleanfp: once more, with the lines kept, the clean sounds on which a line passed the first rule)
import sys, os, json, glob, zlib, numpy as np, soundfile as sf
os.chdir('/tmp/claude-0/v2/sweep'); sys.path.insert(0, '/tmp/claude-0/v2/sweep'); sys.path += ['/tmp/claude-0/v2/proto', '/tmp/claude-0/v2/bench']
import sweep, score
import sweep as lfo, pickle
part = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); out = 'stats_%s.jsonl' % part
done = {json.loads(l)['id'] for l in open(out)} if os.path.exists(out) else set(); fo = open(out, 'a')
def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000; return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def num(f): return {k: (float(v) if not isinstance(v, str) else v) for k, v in f.items() if np.isscalar(v)}
def flange(x, rng):
    n = len(x); p = dict(lo=float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))), depth=float(rng.uniform(1.0, 6.0)), rate=float(np.exp(rng.uniform(np.log(0.1), np.log(1.2)))), kind=int(rng.integers(0, 2)),
                         g=float(rng.choice([-1, 1]) * rng.uniform(0.4, 0.95)), fb=float(rng.uniform(-0.6, 0.6)), phase=float(rng.uniform(0, 1)), stereo=float(rng.choice([0.0, 0.25, 0.5])))
    y = np.zeros_like(x); t = np.arange(n) / 48000.0
    for c in range(2):
        d = (p['lo'] + p['depth'] * lfo.lfo_shape(p['rate'] * t + p['phase'] + c * p['stereo'], p['kind'])) * 48.0
        y[:, c] = sweep._run(np.ascontiguousarray(x[:, c]), d, p['g'], p['fb'], False)
    return y, p
if part == 'train': files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/*.flac'))[::every]
else: files = sorted(glob.glob('/tmp/claude-0/v2/corpus/dry/*.wav'))[::every]
if part == 'cleanfp':
    keep = set()
    for l in open('stats_clean.jsonl'):
        r = json.loads(l)
        if r['found'] or any(f.get('g', 0) != 0 and sweep.accept(f) for ch in r['cands'] for f in ch): keep.add(r['id'])
    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in keep]
LP = 'lines_%s.pkl' % part; lines = pickle.load(open(LP, 'rb')) if os.path.exists(LP) else {}
for fp in files:
    name = os.path.splitext(os.path.basename(fp))[0]
    if name in done: continue
    x = rd(fp)
    if not np.abs(x).max() > 1e-4: continue
    row = dict(id=name, kind=meta.get(name, {}).get('kind', 'train'), split=meta.get(name, {}).get('split', 'train'))
    if part == 'synth':
        y, p = flange(x, np.random.default_rng(zlib.crc32(name.encode()))); row['par'] = p
    else: y = x
    info = []; F = sweep.find(y, info=info); row['found'] = F is not None; row['cands'] = [[num(f) for f in ch] for ch in info]
    lines[name] = [[dict(d=f['d'].astype(np.float32), w=f['weight'].astype(np.float32), path=f['path'].astype(np.int16)) if f.get('g', 0) != 0 else None for f in ch] for ch in info]
    if part == 'synth':          # what taking each followed-up line out of its channel alone would score, whatever the rule says (for judging the rule)
        for ch, cl in enumerate(info):
            for k, f in enumerate(cl):
                if f.get('g', 0) == 0: continue
                o1 = sweep._run(np.ascontiguousarray(y[:, ch]), sweep.per_sample(f['d'], len(y)), f['g'], f['fb'], True)
                row['cands'][ch][k]['out_alone'] = score.removed(x[:, [ch, ch]], y[:, [ch, ch]], np.stack([o1, o1], 1))[0]; row['cands'][ch][k]['accepted'] = bool(sweep.accept(f))
                tt = np.arange(len(f['d'])) * 256 / 48000.0; td = (p['lo'] + p['depth'] * sweep.lfo_shape(p['rate'] * tt + p['phase'] + ch * p['stereo'], p['kind'])) * 48.0
                row['cands'][ch][k]['d_err'] = float(np.median(np.abs(f['d'] - td)[f['weight'] > 0.05])) if (f['weight'] > 0.05).any() else -1.0
    if F is not None:
        o = sweep.apply(y, F, -1.0); row['g'] = [None if f is None else f['g'] for f in F['ch']]
        if part == 'synth': row['out'] = score.removed(x, y, o)[0]
        else: row['harm_db'] = score.distance(x, o)          # a false report: how far taking it "out" moves the sound (mean dB over the score's pictures)
    fo.write(json.dumps(row) + '\n'); fo.flush(); pickle.dump(lines, open(LP, 'wb'))
