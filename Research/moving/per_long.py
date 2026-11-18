# The same for the long test sounds (/dev/shm/mvl): the found curve of the longest stretch and the effect's settings.
import sys, os, json, numpy as np, torch, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/moving']
import moving
torch.set_num_threads(3); model = moving.model_of('/tmp/claude-0/v2/moving/mv_d.pt'); R = []; M = '/dev/shm/mvl/'
for c in json.load(open(M + 'list.json')):
    if c['kind'] == 'clean': continue
    x = sf.read(M + c['name'] + '_wet.wav', dtype='float64', always_2d=True)[0]; F = moving.find(model, x)
    if not F['found']: print(c['name'], 'not found', flush=True); continue
    a, b = max(F['spans'], key=lambda q: q[1] - q[0]); G = F['raw'][0][a:b] + F['raw'][1][a:b]; np.save('/dev/shm/per/%s.npy' % c['name'], G)
    R.append(dict(name=c['name'], fam=c['kind'], tool=c['tool'], params=c['params'], kind=F['kind'], sure=F['sure'], period=F['period'], a=a, b=b)); print(c['name'], c['tool'], F['kind'], '%.2f' % F['sure'], 'stretch %d frames, period %.1f frames = %.2f s' % (b - a, F['period'], F['period'] / 46.875), flush=True)
    json.dump(R, open('/dev/shm/per/list_long.json', 'w'))
