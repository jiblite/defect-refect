# How much is there to win UNDER the sound at all? Answers that know the truth, rendered the engine's way (one gain per
# band and frame), scored over the whole file, the sound itself (first 6 s) and the ring-out, on the scoreboard's 120
# reverb cases:
#   right      the right answer per cell: the direct part's level against the sound's (what the model is trained on)
#   power      from the POWERS of the direct part and of the reverb part alone, S / (S + R): what is left when the luck
#              of how the two add up in a cell (they can cancel or double) is unknown
#   power k    the same from powers averaged over k frames in time (a listener who knows both parts only roughly)
#   deep n     the right answer, but only the cells where it is n dB or more (the plain cases), nothing elsewhere
#   NP=4 python3 body_ceiling.py [step]    -> body_ceiling.json
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, render, predict, depth2_test as d2, body_diag as bd
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_x+unet_xp2_e5'); SR = 48000
def box(a, k): return a if k <= 1 else cm._box(a, k, 0)
def masks(ft, g):
    LE, LS, LR = ft['LE'], ft['LS'], ft['LR']; S = 10 ** (LS / 10.0); R = 10 ** (LR / 10.0); right = np.clip(LS - LE, -30.0, 0.0); out = [('answer', g), ('right', right)]
    for k in (1, 3, 9, 27): out.append(('power' if k == 1 else 'power %d' % k, np.clip(10 * np.log10(np.maximum(box(S, k), 1e-30) / np.maximum(box(S, k) + box(R, k), 1e-30)), -30.0, 0.0)))
    for n in (3, 6, 12): out.append(('deep %d' % n, np.where(right <= -n, right, 0.0)))
    out.append(('answer, never over right', np.maximum(g, right))); out.append(('answer where right is 6+', np.where(right <= -6, g, 0.0)))
    return out
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; n = len(wet); L = 6 * SR
    ft = cm.features(wet, direct); g = predict.gains(ft, MODEL); Cw = d2.cells(direct, wet); base = [d2.seg(Cw, 0, n), d2.seg(Cw, 0, L), d2.seg(Cw, L, n)]
    r = dict(name=c['name'], split=c['meta']['split'], drr=float(10 * np.log10((direct ** 2).sum() / max((rev ** 2).sum(), 1e-20))), base=base, sc={}, band={})
    for nm, m in masks(ft, g):
        C = d2.cells(direct, render.render(wet, m, 0.0)); r['sc'][nm] = [100.0 * (1.0 - d2.seg(C, a, b) / max(w, 1e-9)) for (a, b), w in zip(((0, n), (0, L), (L, n)), base)]
        r['band'][nm] = [p[0] if p[0] is not None else 0.0 for p in bd.parts(m, ft['LE'], ft['LS'])]
    return r
def show(rows):
    drr = np.array([r['drr'] for r in rows]); names = list(rows[0]['sc'])
    for key, kn in (('sc', 'AS HEARD (the scoreboard measure)'), ('band', 'IN THE BAND PICTURE (what the model is trained on)')):
        print(kn)
        for part, pn in enumerate(('whole file', 'the sound itself', 'the ring-out')):
            print(' %s' % pn); print('  %-26s | %-30s | %-30s | %-30s' % ('', 'all (n=%d)' % len(rows), 'heavy (n=%d)' % (drr < 5).sum(), 'light, 5 dB and more (n=%d)' % (drr >= 5).sum()))
            for nm in names:
                S = np.array([r[key][nm][part] for r in rows]); print('  %-26s | %s' % (nm, ' | '.join('med %5.1f mean %6.1f worse %3.0f%%' % (np.median(S[s]), S[s].mean(), 100 * (S[s] < 0).mean()) for s in (np.ones(len(rows), bool), drr < 5, drr >= 5))))
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--show': show(json.load(open('/tmp/claude-0/v2/train/body_ceiling.json'))); sys.exit(0)
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '1'))) as p: rows = p.map(one, ds, chunksize=1)
    json.dump(rows, open('/tmp/claude-0/v2/train/body_ceiling.json', 'w')); show(rows); print('%.0f s' % (time.time() - t0))
