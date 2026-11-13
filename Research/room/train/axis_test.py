# Does the left/right-aware rendering (axis.py) take more reverb out than the plain one? With the right answer and with a model's.
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, score, render, axis, predict
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_b9')
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; ft = cm.features(wet, direct); gt = np.clip(ft['LS'] - ft['LE'], -30.0, 0.0); gm = predict.gains(ft, MODEL)
    sc0 = score.Scorer(direct); dw = sc0(wet); sc = lambda y: 100.0 * (1.0 - sc0(y) / max(dw, 1e-9)); r = dict(name=c['name'], base=float(dw), drr=float(10 * np.log10((direct ** 2).sum() / max((rev ** 2).sum(), 1e-20))))
    X = cm.fwd(wet)
    for lab, g in (('right answer', gt), ('model', gm)):
        sp = axis.spatial(X, g); r[lab + ' plain'] = sc(render.render(wet, g, 0.0)); r[lab + ' two directions'] = sc(axis.render_axis(wet, g, 0.0, sp=sp)); r[lab + ' plain half'] = sc(render.render(wet, g, 0.5)); r[lab + ' two directions half'] = sc(axis.render_axis(wet, g, 0.5, sp=sp))
        mid = (cm.FC41 > 450) & (cm.FC41 < 7000); r[lab + ' dq'] = float(np.median(sp[1][mid, 1])); r[lab + ' rq'] = float(np.median(sp[1][mid, 3]))
    spT = axis.spatial(X, gt); r['model answer, right directions'] = sc(axis.render_axis(wet, gm, 0.0, sp=spT))
    return r
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 6]
    with Pool(int(os.environ.get('NP', '1'))) as p: rows = p.map(one, ds)
    json.dump(rows, open('/tmp/claude-0/v2/train/axis_test_%s.json' % MODEL, 'w'))
    for k in [k for k in rows[0] if k not in ('name', 'base', 'drr')]:
        v = np.array([r[k] for r in rows]); print('%-36s median %6.2f  mean %6.2f' % (k, np.median(v), np.mean(v)) + ('  pooled %5.1f  below zero %2.0f%%' % (100.0 * (1.0 - sum(r['base'] * (1.0 - r[k] / 100.0) for r in rows) / sum(r['base'] for r in rows)), 100 * np.mean(v < 0)) if not k.endswith(('dq', 'rq')) else ''))
    for lab in ('right answer', 'model'):
        dlt = np.array([r[lab + ' two directions'] - r[lab + ' plain'] for r in rows]); print('%s: two directions against plain, per case: median %+.1f, better on %.0f%%, quarter points %+.1f %+.1f, worst %+.1f, best %+.1f' % (lab, np.median(dlt), 100 * np.mean(dlt > 0), np.percentile(dlt, 25), np.percentile(dlt, 75), dlt.min(), dlt.max()))
    print('%.0f s' % (time.time() - t0))
