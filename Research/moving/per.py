# How well is the curve's period read? Every phaser and filter sweep sound of the scoreboard: the found curve of the longest
# stretch (kept in /dev/shm/per) and the effect's settings.
#   python3 per.py
import sys, os, json, numpy as np, torch
sys.path[:0] = ['/tmp/claude-0/v2/moving', '/tmp/claude-0/v2/sweep']
import moving, pic
from lab import cases, load
torch.set_num_threads(3); model = moving.model_of('/tmp/claude-0/v2/moving/mv_d.pt'); R = []
for fam in ('filtersweep', 'phaser'):
    for d in cases(fam):
        meta = json.load(open(d + 'meta.json'))
        for w in 'AB':
            c = load(d, w); F = moving.find(model, c['wet']); name = c['name'] + '_' + w
            if not F['found']: print(name, 'not found', flush=True); continue
            a, b = max(F['spans'], key=lambda q: q[1] - q[0]); G = F['raw'][0][a:b] + F['raw'][1][a:b]; np.save('/dev/shm/per/%s.npy' % name, G)
            R.append(dict(name=name, fam=fam, tool=meta['effect'], params=meta['params'], kind=F['kind'], sure=F['sure'], period=F['period'], a=a, b=b)); print(name, F['kind'], '%.2f' % F['sure'], 'period %.1f frames = %.2f s' % (F['period'], F['period'] / 46.875), json.dumps(meta['params'])[:200], flush=True)
json.dump(R, open('/dev/shm/per/list.json', 'w'))
