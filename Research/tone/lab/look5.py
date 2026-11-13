# Which long-term statistic of a clean sound is smoothest along frequency (smallest own fine detail)? The smoother, the better a filter shows on it.
import sys, glob, json, time
import numpy as np, soundfile as sf, scipy.ndimage as ndi
import feat, model
def stats(x):
    P = feat.frames(x); a = feat.active(P); Q = (P[0] + P[1])[a]; top = np.percentile(Q, 95, axis=0); fl = np.maximum(top * 1e-4, 1e-30); L = 10 * np.log10(np.maximum(Q, fl))
    o = {}
    o['lmean'] = feat.to_grid(L.mean(0)); o['p10'] = feat.to_grid(np.percentile(L, 10, 0)); o['p90'] = feat.to_grid(np.percentile(L, 90, 0)); o['pmean'] = feat.to_grid(10 * np.log10(Q.mean(0) + 1e-30))
    for k in (5, 9, 15):
        o['med%d' % k] = feat.to_grid(ndi.median_filter(L, (1, k), mode='nearest').mean(0))
    o['lowq13'] = feat.to_grid(ndi.percentile_filter(L, 25, (1, 13), mode='nearest').mean(0))
    o['max9'] = feat.to_grid(ndi.maximum_filter(L, (1, 9), mode='nearest').mean(0)); o['max25'] = feat.to_grid(ndi.maximum_filter(L, (1, 25), mode='nearest').mean(0))
    # log of band power per frame (cells of 1/24 octave), mean over frames, each cell clamped 40 dB under its own loud frames
    cs = np.concatenate([np.zeros((len(Q), 1)), np.cumsum(Q, 1)], 1); lo, hi = feat._lo, feat._hi; B = (cs[:, np.maximum(hi, lo + 1)] - cs[:, lo]) / np.maximum(hi - lo, 1)
    tb = np.percentile(B, 95, axis=0); LB = 10 * np.log10(np.maximum(B, np.maximum(tb * 1e-4, 1e-30))); v = LB.mean(0); v[~feat._wide] = o['lmean'][~feat._wide]; o['bandlog'] = v
    dL = np.diff(L, axis=0); on = np.maximum(dL, 0).sum(1); k = max(1, int(round(0.2 * len(on)))); sel = np.argsort(on)[-k:] + 1; o['onset'] = feat.to_grid(L[sel].mean(0))
    lev = feat.to_grid(10 * np.log10(top + 1e-30)); return o, lev
files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac'))[::5]; kinds = json.load(open('/tmp/claude-0/v2/train/dry_meta.json'))
regs = [(100, 300), (300, 1000), (1000, 3000), (3000, 10000)]; acc = {}; t0 = time.time()
for p in files:
    x = sf.read(p, always_2d=True)[0].astype(np.float64); o, lev = stats(x); aud = lev > lev.max() - 50
    for k, v in o.items():
        fine = v - model.smooth12(v)
        for j, (a, b) in enumerate(regs):
            r = aud & (feat.GRID >= a) & (feat.GRID < b)
            if r.sum() > 8: acc.setdefault((k, j), []).append(np.sqrt((fine[r] ** 2).mean()))
print('%d clean clips, %.0f s. Fine detail (rms dB around a half-octave trend) of each statistic, mean over clips [median]:' % (len(files), time.time() - t0))
print('%-9s' % '', ' '.join('%14s' % ('%d-%d Hz' % r) for r in regs))
for k in o: print('%-9s' % k, ' '.join('%7.2f [%4.2f]' % (np.mean(acc[(k, j)]), np.median(acc[(k, j)])) for j in range(len(regs))))
