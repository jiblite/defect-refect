# Adds to the rows of sure_stats_<MODEL>.json how much of each sound is NOT shared between left and right ("own", dB under
# the whole: the energy-weighted unshared share at 256 ms over the cells within 50 dB of the loudest).
#   MODEL=<name> python3 own_stats.py
import sys, os, json
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth
MODEL = os.environ.get('MODEL', 'unet_b9'); P = '/tmp/claude-0/v2/train/sure_stats_%s.json' % MODEL; rows = json.load(open(P))
def own(x):
    ft = cm.features(x); LE = ft['LE']; live = LE > LE.max() - 50.0; E = 10 ** (LE / 10); u = 10 ** (np.clip(ft['U256'], -40, 0) / 10)
    return float(10 * np.log10(max((E * u)[live].sum(), 1e-30) / E[live].sum()))
cache = {}
for r in rows:
    nm = r['name']; dry = nm.endswith(' dry'); base = nm[:-4] if dry else nm; case, w = base.rsplit('/', 1); grp = 'tone' if r['what'] == 'tone only' else 'reverb'
    k = (case, w)
    if k not in cache: cache[k] = truth.load('/tmp/claude-0/v2/bench/sets/%s/%s/' % (grp, case), w)
    r['own'] = own(cache[k]['dry' if dry else 'wet'])
json.dump(rows, open(P, 'w'))
q = lambda v: ' '.join('%6.1f' % a for a in np.percentile(v, [0, 10, 25, 50, 75, 90, 100])) if len(v) else '-'
for lab, sel in [('with reverb, room found', lambda r: r['wet'] and r['found']), ('with reverb, NOT found', lambda r: r['wet'] and not r['found']), ('without, room found', lambda r: not r['wet'] and r['found']), ('without, NOT found', lambda r: not r['wet'] and not r['found'])]:
    v = [r['own'] for r in rows if sel(r)]; print('%-26s n=%3d | not shared, dB (min 10%% 25%% 50%% 75%% 90%% max): %s' % (lab, len(v), q(v)))
for r in rows:
    if r['wet'] and not r['found']: print('   not found: %-30s drr %5.1f own %6.1f' % (r['name'], r['drr'], r['own']))
