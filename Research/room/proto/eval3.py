# Room measurement against the exact truth, from the cache. -v prints every case.
import sys, json, glob, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, room
S = '/tmp/claude-0/v2/bench/sets/reverb/'; CA = '/tmp/claude-0/v2/proto/cache/reverb/'
corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
rows = []; t0 = time.time(); W = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in 'AB' else 'A'
for f in sorted(glob.glob(CA + '*.npz')):
    name = os.path.basename(f)[:-4]; m = json.load(open(S + name + '/meta.json')); z = np.load(f)
    tt, tk = z['true_tau'], z[W + '_kappa']; mid = (room.FC > 450) & (room.FC < 7000) & np.isfinite(tt) & np.isfinite(tk)
    if mid.sum() < 3: continue
    r = room.analyse_env(dict(slow=z[W + 'w_slow'].astype(float), fast=z[W + 'w_fast'].astype(float)))
    d_true = np.mean(-10 * np.log10(tk[mid])); t_true = np.exp(np.mean(np.log(13.8155 * tt[mid])))
    if not r['found']: rows.append((name, corp[m[W]]['kind'], d_true, np.nan, np.nan, np.nan, t_true, 0)); continue
    Ef = z[W + 'w_fast'].astype(float); kt = []
    for b in np.where(mid)[0]:
        g, w = room.contrast(Ef[:, b], r['tau'][b])
        if len(g) >= 3:
            o = np.argsort(g); cw = np.cumsum(w[o]) / w.sum(); kt.append(g[o][np.searchsorted(cw, float(os.environ.get('Q', '0.3')))] + np.log(r['tau'][b]))
    d_L = np.mean(r['drr_db'][mid]); d_T = -4.3429 * np.median(kt) if kt else np.nan
    t_ratio = np.exp(np.mean(np.log(r['T60'][mid]))) / t_true
    rows.append((name, corp[m[W]]['kind'], d_true, d_L, d_T, t_ratio, t_true, len(r['segs'])))
if '-v' in sys.argv:
    for r in rows: print('%-24s %-16s DRR true %6.1f | level fit %6.1f (err %5.1f) | transients %6.1f (err %5.1f) | T60 true %.2f, found x%.2f | decays %d' % (r[0], r[1], r[2], r[3], r[3] - r[2], r[4], r[4] - r[2], r[6], r[5], r[7]))
a = np.array([[r[2], r[3], r[4], r[5], r[6]] for r in rows], float)
def stat(sel, col, label):
    e = (a[sel, col] - a[sel, 0]); ok = np.isfinite(e); e = e[ok]
    if len(e) > 2: print('  %-28s n=%3d  median err %5.1f  median |err| %4.1f  |err|<4 dB: %3.0f%%  corr %.2f' % (label, len(e), np.median(e), np.median(np.abs(e)), 100 * np.mean(np.abs(e) < 4), np.corrcoef(a[sel, 0][ok], a[sel, col][ok])[0, 1]))
def tstat(sel, label):
    l = np.log2(a[sel, 3]); l = l[np.isfinite(l)]
    if len(l) > 2: print('  %-28s n=%3d  T60 median x%.2f (bias x%.2f), within 25%%: %.0f%%, within a factor 1.5: %.0f%%' % (label, len(l), 2 ** np.median(np.abs(l)), 2 ** np.median(l), 100 * np.mean(np.abs(l) < np.log2(1.25)), 100 * np.mean(np.abs(l) < np.log2(1.5))))
print('cases', len(rows), ' not found:', int(np.isnan(a[:, 3]).sum()), ' %.0f s' % (time.time() - t0))
al = np.ones(len(a), bool)
for sel, lab in [(al, 'all'), (a[:, 0] >= 5, 'true DRR >= 5 dB (light)'), ((a[:, 0] < 5) & (a[:, 0] >= -2), 'true DRR -2..5 dB'), (a[:, 0] < -2, 'true DRR < -2 dB (wet)'), (a[:, 4] < 0.5, 'T60 < 0.5 s'), ((a[:, 4] >= 0.5) & (a[:, 4] < 2), 'T60 0.5..2 s'), (a[:, 4] >= 2, 'T60 >= 2 s')]:
    print(lab); tstat(sel, 'decay time'); stat(sel, 1, 'level fit'); stat(sel, 2, 'transients')
for k in sorted(set(r[1] for r in rows)):
    sel = np.array([r[1] == k for r in rows]); print(k); tstat(sel, 'decay time'); stat(sel, 1, 'level fit'); stat(sel, 2, 'transients')
