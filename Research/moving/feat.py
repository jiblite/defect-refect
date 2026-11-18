# Measures per sound from the kept readings (/dev/shm/mvr), for the rule study.
import sys, os, glob, json, numpy as np
OUT = '/dev/shm/mvr/'
def feats(path):
    d = np.load(path, allow_pickle=True); fam, name = os.path.basename(path)[:-4].split('__', 1); r = dict(fam=fam, name=name, found=bool(d['found']))
    if not r['found']: return r
    G = d['G'].astype(float); P = d['P'].astype(float); m = d['gate'] > 0; T = len(m); on = np.maximum(1 - P[0][:, 0], 1 - P[1][:, 0])
    r['kind'] = str(d['kind']); r['say'] = float(on[m].mean()); r['share'] = float(m.mean()); r['low'] = float(np.quantile(on[m], 0.1)); r['period'] = float(d['period'])
    Gi = G[:, m]; r['size'] = float(np.sqrt((Gi ** 2).mean())); mean = Gi.mean(1, keepdims=True); r['move'] = float(((Gi - mean) ** 2).mean() / max((Gi ** 2).mean(), 1e-12))
    r['sides'] = float((Gi[0] * Gi[1]).sum() / max(np.sqrt((Gi[0] ** 2).sum() * (Gi[1] ** 2).sum()), 1e-12))
    r['pk'] = float(sum(P[c][m, 1].sum() for c in range(2)) / max(sum(P[c][m, 1:].sum() for c in range(2)), 1e-12))          # share of 'filter sweep' in the say-so
    r['span'] = float(np.abs(Gi).max()); r['depth'] = float(np.mean(Gi.max(2) - Gi.min(2)))
    if 'P2' in d:
        P2 = d['P2'].astype(float); G2 = d['G2'].astype(float); T2 = min(T, P2.shape[1]); m2 = m[:T2]; on2 = np.maximum(1 - P2[0][:T2, 0], 1 - P2[1][:T2, 0])
        r['after'] = float(on2[m2].mean()); r['size2'] = float(np.sqrt((G2[:, :T2][:, m2] ** 2).mean()))
        a = Gi[:, :m2.sum()]; b = G2[:, :T2][:, m2]; r['again'] = float((a * b).sum() / max((a * a).sum(), 1e-12))            # how much of the same curve it reads again
    if 'out' in d: r['out'] = float(d['out'])
    return r
def name_of(r): return r['name'] if 'real' not in r['name'] else 'own-recording'
def line(r):
    return '%-11s %-24s %-11s say %.2f low %.2f share %.2f size %5.2f move %.2f sides %+.2f depth %5.1f | after %.2f size2 %5.2f again %+.2f %s' % (r['fam'], name_of(r)[:24], r['kind'], r['say'], r['low'], r['share'], r['size'], r['move'], r['sides'], r['depth'],
        r.get('after', -1), r.get('size2', -1), r.get('again', 0), ('out %7.1f' % r['out']) if 'out' in r else '')
if __name__ == '__main__':
    R = [feats(p) for p in sorted(glob.glob(OUT + '*.npz'))]; json.dump(R, open('feat_mv_d.json', 'w'))
    fams = sys.argv[1:] or sorted({r['fam'] for r in R})
    for f in fams:
        rows = [r for r in R if r['fam'] == f]; print('== %s: %d sounds, %d found' % (f, len(rows), sum(r['found'] for r in rows)))
        for r in sorted([r for r in rows if r['found']], key=lambda r: -r['say']): print('  ', line(r))
