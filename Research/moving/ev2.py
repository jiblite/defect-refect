# The prototype with its rule on the scoreboard and the clean corpus: what shows at the app's default Sureness, and for the
# phaser and filter sweep cases what taking the curve off and putting it on the case's other sound scores.
#   python3 ev2.py model out.json [family ...]       (no family: phaser, filtersweep, clean and the 17 others)
import sys, os, json, numpy as np, torch
sys.path[:0] = ['/tmp/claude-0/v2/moving', '/tmp/claude-0/v2/sweep']
import moving, pic, soundfile as sf
from lab import cases, load, score
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
SETS = '/tmp/claude-0/v2/bench/sets/'
model = moving.model_of('/tmp/claude-0/v2/moving/%s.pt' % sys.argv[1]); fams = sys.argv[3:] or ['phaser', 'filtersweep', 'clean'] + sorted(f for f in os.listdir(SETS) if f not in ('filtersweep', 'phaser')); R = []
def row(fam, name, F, **kw): R.append(dict(fam=fam, name=name, found=F['found'], kind=F['kind'], sure=F['sure'], say=F['say'], size=F['size'], reads=F['reads'], share=float((F['gate'] > 0).mean()), spans=len(F['spans']), named=len(F['all']), **kw))
for fam in fams:
    if fam == 'clean':
        meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
        for cid in sorted(meta):
            x = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % cid, dtype='float64', always_2d=True)[0]; x = x if x.shape[1] > 1 else np.repeat(x, 2, 1); row('clean', cid, moving.find(model, x[:, :2]))
    else:
        for d in cases(fam):
            for w in 'AB':
                c = load(d, w); F = moving.find(model, c['wet']); kw = {}
                if fam in ('phaser', 'filtersweep') and F['found']:
                    o = load(d, 'B' if w == 'A' else 'A'); kw['out'] = score.removed(c['dry'], c['wet'], moving.apply(c['wet'], F, -1.0))[0]; kw['onto'] = score.removed(o['wet'], o['dry'], moving.apply(o['dry'], F, 1.0, own=False))[0]
                    print('%-20s %s %-11s sure %.2f (say %.2f, size %5.2f dB, read %d times) | out %7.1f onto %7.1f%s' % (c['name'], w, F['kind'], F['sure'], F['say'], F['size'], F['reads'], kw['out'], kw['onto'], '' if F['sure'] > moving.CUT else '   (not shown)'), flush=True)
                row(fam, c['name'] + '_' + w, F, **kw)
    rows = [r for r in R if r['fam'] == fam]; on = [r for r in rows if r['found'] and r['sure'] > moving.CUT]
    print('%-12s %3d sounds | shown %3d (sweep %d, phaser %d)%s' % (fam, len(rows), len(on), sum(r['kind'] == 'filtersweep' for r in on), sum(r['kind'] == 'phaser' for r in on),
          (' | where shown: out median %.1f, onto median %.1f, worse than nothing %d' % (np.median([r['out'] for r in on]), np.median([r['onto'] for r in on]), sum(r['out'] < 0 for r in on))) if on and 'out' in on[0] else ''), flush=True)
    json.dump(R, open(sys.argv[2], 'w'))
