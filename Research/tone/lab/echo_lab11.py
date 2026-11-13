import sys, os, glob, json, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, soundfile as sf
from multiprocessing import Pool
import feat, echo2
KW = dict(iso=float(sys.argv[4]) if len(sys.argv) > 4 else 0.5, tmin=float(sys.argv[5]) if len(sys.argv) > 5 else 0.25, tlow=float(sys.argv[6]) if len(sys.argv) > 6 else 0.40)
def one(a):
    p, seed = a; x = sf.read(p, always_2d=True)[0].astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    P = feat.frames(x[:, :2]); act = feat.active(P); Q = (P[0] + P[1])[act]; r = np.random.default_rng(seed); out = [echo2.find5(Q, **KW)]
    for g in (0.3, 0.45, 0.7):
        d = float(np.exp(r.uniform(np.log(0.6 * 48), np.log(12 * 48)))); sg = g * float(r.choice([-1, 1])); out.append((d, sg, echo2.find5(Q * np.exp(2 * echo2.comb_ln(d, sg))[None, :], **KW)))
    return out
if __name__ == '__main__':
    if sys.argv[1] == 'dev':
        from base import *
        seen = set(); nfa = 0
        for d in case_dirs():
            m = json.load(open(d + 'meta.json'))
            if m['split'] != 'dev': continue
            for which in ('A', 'B'):
                c = load(d, which); P = feat.frames(c['wet']); act = feat.active(P); fw = echo2.find5((P[0] + P[1])[act], **KW)
                if which == 'A' and (m['effect'] == 'tone_comb' or fw): print('%s wet A: %s%s' % (c['name'], ' '.join('(%.2f, %+.3f)' % t for t in fw) or 'nothing', '   true (%.2f, %+.3f)' % (m['params']['ms'] * 48, m['params']['g']) if m['effect'] == 'tone_comb' else ''))
                if m[which] in seen: continue
                seen.add(m[which]); P = feat.frames(c['dry']); act = feat.active(P); f = echo2.find5((P[0] + P[1])[act], **KW)
                if f: nfa += 1; print('   dry sound of %s/%s: %s' % (c['name'], which, ' '.join('(%.2f, %+.3f)' % t for t in f)))
        print('dev dry sounds: %d distinct, false alarms on %d' % (len(seen), nfa))
    else:
        files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[int(sys.argv[3])::int(sys.argv[2])]; t0 = time.time()
        with Pool(2) as pool: R = pool.map(one, [(p, i) for i, p in enumerate(files)], chunksize=4)
        kind = np.array([os.path.basename(p).split('_')[1] for p in files]); fa = np.array([len(r[0]) > 0 for r in R])
        print('clean clips %d: false alarms %.1f%% (%s) gains %s' % (len(files), 100 * fa.mean(), ' '.join('%s %.0f%%' % (k[:9], 100 * fa[kind == k].mean()) for k in sorted(set(kind)) if fa[kind == k].any()), np.round(sorted([abs(r[0][0][1]) for r in R if r[0]]), 2)))
        for j, g in enumerate((0.3, 0.45, 0.7)):
            near = [abs(r[j + 1][2][0][0] - r[j + 1][0]) for r in R if len(r[j + 1][2]) > 0 and abs(r[j + 1][2][0][0] - r[j + 1][0]) < 2.0]; other = np.mean([len(r[j + 1][2]) > 0 and abs(r[j + 1][2][0][0] - r[j + 1][0]) >= 2.0 for r in R]); extra = np.mean([len(r[j + 1][2]) > 1 for r in R])
            print('g=%.2f: found %.0f%%, something else first %.0f%%, a second reflection reported %.0f%% | delay error median %.3f, 90th %.3f samples' % (g, 100 * len(near) / len(R), 100 * other, 100 * extra, *np.percentile(near, [50, 90])))
        print('%.0f s' % (time.time() - t0))
