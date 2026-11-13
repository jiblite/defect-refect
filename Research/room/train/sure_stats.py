# How sure can the app be that a sound has a room on it? Collects, for sounds WITH reverb (the scoreboard's wet sounds) and
# WITHOUT (the same sounds dry, and the tone-only test cases), what the engine's analysis says, for a rule to be fitted on.
#   MODEL=<name> NP=1 python3 sure_stats.py [step]      -> sure_stats_<MODEL>.json
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, engine, roomnum
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_b9')
def stats(x):
    an = engine.analyse(x, MODEL); LE = an['LE']; g = an['g']; live = LE > LE.max() - 50.0; E = 10 ** (LE / 10); G = 10 ** (g / 10)
    s = dict(found=bool(an['found']), share=float(10 * np.log10(max((E * (1 - G))[live].sum(), 1e-30) / max((E * G)[live].sum(), 1e-30))), deep=float(np.mean(g[live] < -12.0)), gmean=float(np.mean(g[live])), frames=int(LE.shape[0]))
    if an['found']:
        mid = roomnum.MID; ev = an['ev']; s.update(kmid=float(np.median(an['kdb'][mid])), tmid=float(np.median(an['T60'][mid])), ev=float(ev.sum()), evmid=float(np.median(ev[mid])), evb=int((ev > 0).sum()), wid=float(np.median(an['width'])))
    return s
def one(d):
    rows = []; tone = '/tone/' in d
    for w in 'AB':
        c = truth.load(d, w); m = c['meta']
        if tone: rows.append(dict(name=c['name'] + '/' + w, wet=0, what='tone only', split=m.get('split', ''), **stats(c['wet'])))
        else:
            direct, rev = truth.parts(c); drr = float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20)))
            rows.append(dict(name=c['name'] + '/' + w, wet=1, what='reverb', drr=drr, split=m.get('split', ''), **stats(c['wet']))); rows.append(dict(name=c['name'] + '/' + w + ' dry', wet=0, what='dry', split=m.get('split', ''), **stats(c['dry'])))
    return rows
if __name__ == '__main__':
    t0 = time.time(); step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::step] + sorted(glob.glob('/tmp/claude-0/v2/bench/sets/tone/*/'))[::step]
    with Pool(int(os.environ.get('NP', '1'))) as p: rows = sum(p.map(one, ds), [])
    json.dump(rows, open('/tmp/claude-0/v2/train/sure_stats_%s.json' % MODEL, 'w'))
    for lab, sel in [('with reverb', [r for r in rows if r['wet']]), ('dry', [r for r in rows if r['what'] == 'dry']), ('tone only', [r for r in rows if r['what'] == 'tone only'])]:
        if sel: print('%-12s n=%3d | a room was found on %3.0f%% | share (dB): quarter points %.1f %.1f %.1f | deep cells: %.2f %.2f %.2f' % (lab, len(sel), 100 * np.mean([r['found'] for r in sel]), *np.percentile([r['share'] for r in sel], [25, 50, 75]), *np.percentile([r['deep'] for r in sel], [25, 50, 75])))
    print('%.0f s' % (time.time() - t0))
