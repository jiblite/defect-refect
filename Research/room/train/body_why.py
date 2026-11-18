# Why is the answer weak under the sound? Per scoreboard reverb case, in the band picture and over the sound itself
# (first 6 s): the points the answer wins, the points it loses by turning down cells that belong to the sound, and how
# much of that loss sits in cells the model ALSO turns down when it is given the dry sound alone (the sound's own ring
# read as a room) against cells it only turns down once the reverb is on.
#   NP=4 python3 body_why.py    -> body_why.json
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, predict
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_x+unet_xp2_e5'); BODY = 562
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; ft = cm.features(wet, direct); g = predict.gains(ft, MODEL); fd = cm.features(direct); gd = predict.gains(fd, MODEL)
    LE, LS = ft['LE'][:BODY], ft['LS'][:BODY]; g = g[:BODY]; gd = gd[:BODY]; top = ft['LE'].max(); fl = top - 60.0; ref = np.maximum(LS, fl); right = np.clip(LS - LE, -30.0, 0.0)
    e0 = np.minimum(np.abs(np.maximum(LE, fl) - ref), 30.0); e1 = np.minimum(np.abs(np.maximum(LE + g, fl) - ref), 30.0); base = e0.sum(); d_ = e0 - e1          # points per cell: + won, - lost
    live = LE > fl; over = live & (g < right - 3.0); own = over & (gd < -3.0)
    ldry = fd['LE'][:BODY]; lived = ldry > fd['LE'].max() - 50.0; Ed = 10 ** ((ldry - ldry.max()) / 10.0)
    return dict(name=c['name'], split=c['meta']['split'], drr=float(10 * np.log10((direct ** 2).sum() / max((rev ** 2).sum(), 1e-20))), base=float(base / e0.size),
                score=float(100 * d_.sum() / base), won=float(100 * d_[d_ > 0].sum() / base), lost=float(-100 * d_[d_ < 0].sum() / base), lost_over=float(-100 * d_[over].sum() / base), lost_own=float(-100 * d_[own].sum() / base),
                missed=float(100 * (e1[live & (g > right + 3.0)]).sum() / base), dry_deep=float(np.mean(gd[lived] < -12.0)), dry_taken=float(10 * np.log10(max((Ed * (1 - 10 ** (gd / 10.0)))[lived].sum(), 1e-30) / Ed[lived].sum())),
                dry_cells3=float(np.mean(gd[lived] < -3.0)))
def show(rows):
    A = lambda k, rs: np.array([r[k] for r in rs]); drr = A('drr', rows)
    print('%d cases, model %s; points of the band score over the sound itself (100 = all of the untouched distance)' % (len(rows), MODEL))
    for lab, sel in (('all', drr > -99), ('heavy (direct < 5 dB over)', drr < 5), ('light (5 dB and more)', drr >= 5), ('light and worse than untouched', (drr >= 5) & (A('score', rows) < 0))):
        rs = [r for r, s in zip(rows, sel) if s]
        if not rs: continue
        print('%-32s n=%3d | score med %5.1f mean %6.1f | won %5.1f | lost %5.1f, of it by going too deep %5.1f, of that in cells the dry sound alone also loses %5.1f (%2.0f%%) | left in (missed) %5.1f | dry sound alone: cells under -3 dB %4.1f%%, under -12 dB %4.1f%%' % (
            lab, len(rs), np.median(A('score', rs)), A('score', rs).mean(), A('won', rs).mean(), A('lost', rs).mean(), A('lost_over', rs).mean(), A('lost_own', rs).mean(), 100 * A('lost_own', rs).sum() / max(A('lost_over', rs).sum(), 1e-9), A('missed', rs).mean(), 100 * A('dry_cells3', rs).mean(), 100 * A('dry_deep', rs).mean()))
    sc = A('score', rows); dd = A('dry_cells3', rows); print('rank correlation of the score with "cells the dry sound alone loses": all %.2f, light %.2f' % (spear(sc, dd), spear(sc[drr >= 5], dd[drr >= 5])))
    o = np.argsort(sc)[:14]; print('the 14 worst:'); [print('  %-34s direct %5.1f dB over | score %7.1f | won %5.1f lost %6.1f (own ring %6.1f) | dry alone under -3 dB: %4.1f%%' % (rows[i]['name'][:34], rows[i]['drr'], rows[i]['score'], rows[i]['won'], rows[i]['lost'], rows[i]['lost_own'], 100 * rows[i]['dry_cells3'])) for i in o]
def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--show': show(json.load(open('/tmp/claude-0/v2/train/body_why.json'))); sys.exit(0)
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))
    with Pool(int(os.environ.get('NP', '1'))) as p: rows = p.map(one, ds, chunksize=1)
    json.dump(rows, open('/tmp/claude-0/v2/train/body_why.json', 'w')); show(rows); print('%.0f s' % (time.time() - t0))
