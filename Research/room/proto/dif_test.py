# Does the "not shared between the channels" power follow the true reverb power? Against the exact truth, per case.
import sys, json, glob, os
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, room, truth
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/reverb/'; corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
mid = (room.FC > 450) & (room.FC < 7000)
def one(d):
    c = truth.load(d); t = truth.room_truth(c); E, D = room.diffuse_power(c['wet']); Ed, Dd = room.diffuse_power(c['dry'])
    direct, rev = truth.parts(c); Er, Dr = room.diffuse_power(rev)
    n = min(len(E), len(t['R'])); R = t['R'][:n, mid]; Sx = t['S'][:n, mid]; Dm = D[:n, mid]; Em = E[:n, mid]
    live = Em > Em.max() * 1e-5
    return (c['name'], corp[c['meta']['A']]['kind'], float(np.nanmean(t['drr_db'][mid])), 10 * np.log10(Dm.sum() / max(R.sum(), 1e-30)), float(np.corrcoef(np.log10(Dm[live] + 1e-12), np.log10(R[live] + 1e-12))[0, 1]),
            10 * np.log10(max(Dd[:, mid].sum(), 1e-30) / Ed[:, mid].sum()), 10 * np.log10(max(Dr[:, mid].sum(), 1e-30) / Er[:, mid].sum()), 10 * np.log10(Dm.sum() / max((Em - Dm).sum(), 1e-30)))
if __name__ == '__main__':
    with Pool(4) as p: rows = p.map(one, sorted(glob.glob(S + '*/')))
    for r in rows: print('%-24s %-16s DRR true %6.1f | unshared power / true reverb power %6.1f dB, tracks agree %.2f | dry sound unshared %6.1f dB | reverb part unshared %6.1f dB | DRR from unshared %6.1f' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], -r[7]))
    a = np.array([[r[2], r[3], r[4], r[5], r[6], -r[7]] for r in rows])
    for lab, sel in [('all', np.ones(len(a), bool)), ('dry sound shared (unshared < -15 dB)', a[:, 3] < -15), ('  and reverb part mostly unshared (> -3 dB)', (a[:, 3] < -15) & (a[:, 4] > -3)), ('dry sound wide (unshared >= -15 dB)', a[:, 3] >= -15)]:
        if sel.sum() > 2: print('%-46s n=%3d  unshared/true reverb: median %5.1f dB, |.|<3 dB in %3.0f%% | DRR estimate: median err %5.1f, median |err| %4.1f, corr %.2f' % (lab, sel.sum(), np.median(a[sel, 1]), 100 * np.mean(np.abs(a[sel, 1]) < 3), np.median(a[sel, 5] - a[sel, 0]), np.median(np.abs(a[sel, 5] - a[sel, 0])), np.corrcoef(a[sel, 5], a[sel, 0])[0, 1]))
