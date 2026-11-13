# The real scoreboard number for a mask model: de-reverb every scoreboard reverb case and score against the true direct part.
import sys, os, json, glob, pickle, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, common as cm, truth, score, render, gbfeat
from multiprocessing import Pool
corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); MODEL = None
def predict(LE, U64, U256):
    global MODEL
    if MODEL is None: MODEL = pickle.load(open(os.environ.get('MODEL', '/tmp/claude-0/v2/train/gb.pkl'), 'rb'))
    X = gbfeat.features(LE, U64, U256); return np.clip(MODEL.predict(X.reshape(-1, X.shape[-1])).reshape(LE.shape), -30.0, 0.0)
def one(a):
    d, w = a; c = truth.load(d, w); direct, rev = truth.parts(c); wet = c['wet']; ft = cm.features(wet, direct); sc0 = score.Scorer(direct); dw = sc0(wet); out = {}
    def sc(y): return 100.0 * (1.0 - sc0(y) / max(dw, 1e-9))
    g = predict(ft['LE'], ft['U64'], ft['U256']); out['model'] = sc(render.render(wet, g, 0.0)); out['model, half way'] = sc(render.render(wet, g, 0.5))
    gt, live = gbfeat.target(ft['LE'], ft['LS']); out['true band gains'] = sc(render.render(wet, gt, 0.0))
    # doubling: the true target is direct + 2 x reverb
    tgt = direct + 2.0 * rev; sc2 = score.Scorer(tgt); d2 = sc2(wet); out['doubled'] = 100.0 * (1.0 - sc2(render.render(wet, g, 2.0)) / max(d2, 1e-9)); out['doubled, true gains'] = 100.0 * (1.0 - sc2(render.render(wet, gt, 2.0)) / max(d2, 1e-9))
    m = c['meta']; ed = (direct ** 2).sum(); er = (rev ** 2).sum()
    return c['name'] + '_' + w, m['split'], corp[m[w]]['kind'], float(10 * np.log10(max(ed, 1e-20) / max(er, 1e-20))), dw, out
if __name__ == '__main__':
    t0 = time.time(); jobs = [(d, w) for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/')) for w in (sys.argv[1] if len(sys.argv) > 1 else 'A')]
    with Pool(4) as p: rows = p.map(one, jobs)
    keys = list(rows[0][5].keys()); a = np.array([[r[3], r[4]] + [r[5][k] for k in keys] for r in rows]); sp = np.array([r[1] for r in rows])
    if '-v' in sys.argv:
        for r in rows: print('%-26s %-5s %-16s DRR %6.1f  wet is %4.1f dB from dry | ' % r[:5] + ' | '.join('%s %6.1f' % (k, r[5][k]) for k in keys))
    for lab, sel in [('reverbs also used in training', sp == 'dev'), ('reverbs and sounds never seen in training', sp == 'test')]:
        for lab2, s2 in [('all', np.ones(len(a), bool)), ('light (DRR >= 8)', a[:, 0] >= 8), ('middle', (a[:, 0] < 8) & (a[:, 0] >= 0)), ('wet (DRR < 0)', a[:, 0] < 0)]:
            s = sel & s2
            if s.sum(): print('%-44s %-18s n=%3d | ' % (lab if lab2 == 'all' else '', lab2, s.sum()) + ' | '.join('%s: median %5.1f, below zero %2.0f%%' % (k, np.median(a[s, 2 + i]), 100 * np.mean(a[s, 2 + i] < 0)) for i, k in enumerate(keys)))
    json.dump([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], open(os.environ.get('OUT', '/tmp/claude-0/v2/train/sb_full.json'), 'w')); print('%.0f s' % (time.time() - t0))
