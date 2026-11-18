# Everything the moving-tone model says on every sound of the scoreboard and the clean corpus, kept per sound for the rule study:
# the curve and the three probabilities per frame and side, and the same again after the found curve is taken off.
#   python3 rule2.py model [family ...]        -> /dev/shm/mvr/<family>__<name>.npz
import sys, os, json, numpy as np
sys.path[:0] = ['/tmp/claude-0/v2/moving', '/tmp/claude-0/v2/sweep']
import torch, soundfile as sf, moving, pic
from lab import cases, load, score
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
SETS = '/tmp/claude-0/v2/bench/sets/'; OUT = '/dev/shm/mvr/'
model = moving.model_of('/tmp/claude-0/v2/moving/%s.pt' % sys.argv[1]); fams = sys.argv[2:] or ['filtersweep', 'phaser', 'clean'] + sorted(f for f in os.listdir(SETS) if f not in ('filtersweep', 'phaser'))

def reading(x):
    same = np.array_equal(x[:, 0], x[:, 1]); r = [moving.read(model, x[:, 0])]; r.append(r[0] if same else moving.read(model, x[:, 1]))
    return np.stack([g for g, _ in r]).astype(np.float32), np.stack([p for _, p in r]).astype(np.float32)

def one(fam, name, x, dry=None):
    path = OUT + '%s__%s.npz' % (fam, name)
    if os.path.exists(path): return
    G, P = reading(x); F = moving.find(model, x); d = dict(G=G, P=P, found=F['found'], kind=F['kind'], sure=F['sure'], gate=F['gate'].astype(np.float32), spans=np.array(F['spans'], dtype=np.int64).reshape(-1, 2), period=F['period'])
    if F['found']:
        out = moving.apply(x, F, -1.0); d['G2'], d['P2'] = reading(out)
        if dry is not None: d['out'] = score.removed(dry, x, out)[0]
    np.savez(path, **d)

for fam in fams:
    if fam == 'clean':
        meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
        for cid in sorted(meta):
            x = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % cid, dtype='float64', always_2d=True)[0]; x = x if x.shape[1] > 1 else np.repeat(x, 2, 1); one('clean', cid, x[:, :2])
    else:
        for d in cases(fam):
            for w in 'AB': c = load(d, w); one(fam, c['name'] + '_' + w, c['wet'], c['dry'] if fam in ('filtersweep', 'phaser') else None)
    print(fam, len(os.listdir(OUT)), flush=True)
