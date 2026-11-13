import sys, json, glob, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, room
S = '/tmp/claude-0/v2/bench/sets/reverb/'; CA = '/tmp/claude-0/v2/proto/cache/reverb/'
corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
rows = []; t0 = time.time()
for f in sorted(glob.glob(CA + '*.npz')):
    name = os.path.basename(f)[:-4]; m = json.load(open(S + name + '/meta.json')); z = np.load(f)
    tt, tA = z['true_tau'], z['true_A']; mid = (room.FC > 450) & (room.FC < 7000) & np.isfinite(tt) & np.isfinite(tA)
    if mid.sum() < 3: continue
    r = room.analyse_env(dict(slow=z['Aw_slow'].astype(float), fast=z['Aw_fast'].astype(float)))
    if not r['found']: rows.append((name, corp[m['A']]['kind'], np.mean(-10 * np.log10(tA[mid] * tt[mid])), np.nan, np.nan, np.nan, np.nan, 0)); continue
    Ef = z['Aw_fast'].astype(float); kt = []; nev = 0
    for b in np.where(mid)[0]:
        g, w = room.contrast(Ef[:, b], r['tau'][b]); nev += len(g)
        if len(g) >= 3:
            o = np.argsort(g); cw = np.cumsum(w[o]) / w.sum(); kt.append(g[o][np.searchsorted(cw, float(os.environ.get('Q', '0.3')))] + np.log(r['tau'][b]))
    d_true = np.mean(-10 * np.log10(tA[mid] * tt[mid])); d_L = np.mean(r['drr_db'][mid]); d_T = -4.3429 * np.median(kt) if kt else np.nan
    t_ratio = np.exp(np.mean(np.log(r['T60'][mid]))) / np.exp(np.mean(np.log(13.8155 * tt[mid])))
    rows.append((name, corp[m['A']]['kind'], d_true, d_L, d_T, t_ratio, len(kt), nev))
if '-v' in sys.argv:
    for r in rows: print('%-24s %-16s DRR true %6.1f | level fit %6.1f (err %5.1f) | transients %6.1f (err %5.1f) | T60 x%.2f | bands %d events %d' % (r[0], r[1], r[2], r[3], r[3] - r[2], r[4], r[4] - r[2], r[5], r[6], r[7]))
a = np.array([[r[2], r[3], r[4], r[5]] for r in rows], float)
def stat(sel, col, label):
    e = (a[sel, col] - a[sel, 0]); e = e[np.isfinite(e)]
    if len(e): print('  %-34s n=%3d  median err %5.1f  median |err| %4.1f  |err|<4 dB: %3.0f%%' % (label, len(e), np.median(e), np.median(np.abs(e)), 100 * np.mean(np.abs(e) < 4)))
wetter = a[:, 0] < -2; drier = a[:, 0] >= -2
print('cases', len(rows), ' T60 median x%.2f, within 25%%: %.0f%%' % (2 ** np.nanmedian(np.abs(np.log2(a[:, 3]))), 100 * np.nanmean(np.abs(np.log2(a[:, 3])) < np.log2(1.25))), ' %.0f s' % (time.time() - t0))
for sel, lab in [(np.ones(len(a), bool), 'all'), (drier, 'true DRR >= -2 dB (drier)'), (wetter, 'true DRR < -2 dB (wetter)')]:
    print(lab); stat(sel, 1, 'level fit'); stat(sel, 2, 'transients')
kinds = sorted(set(r[1] for r in rows))
for k in kinds:
    sel = np.array([r[1] == k for r in rows]); print(k); stat(sel, 1, 'level fit'); stat(sel, 2, 'transients')
