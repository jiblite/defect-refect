# Audio scores of the blind curve on the dev cases (or any split), for several settings at once: parts on/off, restraint, linked/dual.
import sys, os, json, time
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/tmp/claude-0/v2/tone')
import numpy as np
from multiprocessing import Pool
from base import *
import tone
def configs():
    c = {}
    c['echo only'] = dict(echo=True, model=False, th=0.0, st='linked')
    for th in (0.0, 2.0): c['model only, restraint %g' % th] = dict(echo=False, model=True, th=th, st='linked')
    for th in (0.0, 1.0, 2.0, 3.0, 5.0): c['echo + model, restraint %g' % th] = dict(echo=True, model=True, th=th, st='linked')
    c['echo + model, restraint 2, all sizes'] = dict(echo=True, model=True, th=2.0, st='linked', dz=0.0)
    c['dual: echo + model, restraint 2'] = dict(echo=True, model=True, th=2.0, st='dual')
    return c
def curves_for(x, cfgs):
    """All configurations' curves for one sound, analysing each view once."""
    P = tone._frames(x); a = tone._active(P); P = P[:, a, :]; views = dict(linked=[P[0] + P[1]], dual=[P[0], P[1]]); parts = {}; out = {}
    for name, c in cfgs.items():
        key = (c['st'], c['echo'], c['model'])
        if key not in parts: parts[key] = [tone._analyse_view(Q, c['echo'], c['model']) for Q in views[c['st']]]
        cv = np.stack([tone._finish_view(dict(p), c['th'], c.get('dz', tone.MODEL_MIN_DB)) for p in parts[key]]); out[name] = cv[0] if c['st'] == 'linked' else cv
    return out
def one(d):
    cfgs = configs(); a = load(d, 'A'); b = load(d, 'B'); sa = score.Scorer(a['dry']); sb = score.Scorer(b['wet']); sbd = score.Scorer(b['dry']); dwa = sa(a['wet']); dwb = sb(b['dry'])
    row = dict(name=a['name'], effect=a['meta']['effect'], split=a['meta']['split'], dwa=dwa, dwb=dwb, res={})
    cw = curves_for(a['wet'], cfgs); cda = curves_for(a['dry'], cfgs); cdb = curves_for(b['dry'], cfgs)
    for name in cfgs:
        rm = 100 * (1 - sa(tone.apply_curve(a['wet'], tone.FB, cw[name], 0.0)) / dwa); rf = 100 * (1 - sb(tone.put_curve(b['dry'], tone.FB, cw[name], 1.0)) / dwb)
        ha = sa(tone.apply_curve(a['dry'], tone.FB, cda[name], 0.0)); hb = sbd(tone.apply_curve(b['dry'], tone.FB, cdb[name], 0.0)); row['res'][name] = (rm, rf, ha, hb)
    return row
def show(rows, split):
    rows = [r for r in rows if r['split'] == split]; effs = sorted(set(r['effect'] for r in rows)); names = list(rows[0]['res'])
    print('split %s, %d cases. Removed %% / Refect %%: median per effect, then all: median, mean, share below zero. Harm: d(dry, flatten(dry)), mean [90th percentile, max] over the %d dry sounds (typical d(dry, wet): median %.2f)' % (split, len(rows), 2 * len(rows), np.median([r['dwa'] for r in rows])))
    print('%-34s' % '', ' '.join('%11s' % e.replace('tone_', '')[:11] for e in effs), '| all: med  mean  <0 | harm')
    for nm in names:
        for j, lab in ((0, 'rm'), (1, 'rf')):
            v = np.array([r['res'][nm][j] for r in rows]); s = '%-31s %s' % (nm if j == 0 else '', lab)
            for e in effs: s += ' %11.1f' % np.median([r['res'][nm][j] for r in rows if r['effect'] == e])
            s += ' | %6.1f %6.1f %3.0f%%' % (np.median(v), v.mean(), 100 * (v < 0).mean())
            if j == 0: h = np.array([r['res'][nm][k] for r in rows for k in (2, 3)]); s += ' | %.3f [%.3f, %.3f]' % (h.mean(), np.percentile(h, 90), h.max())
            print(s)
if __name__ == '__main__':
    tag = sys.argv[1]; split = sys.argv[2] if len(sys.argv) > 2 else 'dev'; t0 = time.time()
    ds = [d for d in case_dirs() if json.load(open(d + 'meta.json'))['split'] == split]
    with Pool(2) as pool: rows = pool.map(one, ds, chunksize=1)
    json.dump(rows, open(WORK + 'evaldev_%s.json' % tag, 'w')); show(rows, split); print('%.0f s' % (time.time() - t0))
