import sys, json, glob, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, soundfile as sf
import room
S = '/tmp/claude-0/v2/bench/sets/reverb/'; C = '/tmp/claude-0/v2/corpus/dry/'
def rd(p):
    x, sr = sf.read(p, always_2d=True); return x.astype(np.float64)
rows = []; t0 = time.time()
only = sys.argv[1] if len(sys.argv) > 1 else ''
for d in sorted(glob.glob(S + '*/')):
    m = json.load(open(d + 'meta.json'))
    if only and only not in m['effect']: continue
    wet = rd(d + 'A_wet.flac'); ir = rd(d + 'ir.flac'); tr = room.truth_from_ir(ir); r = room.analyse(wet)
    dryr = room.analyse(np.vstack([rd(C + m['A'] + '.wav'), np.zeros((int(m['tail'] * 48000), 2))]))
    mid = (room.FC > 250) & (room.FC < 6000) & np.isfinite(tr['tau'])
    if mid.sum() == 0 or not r['found']:
        rows.append((m['effect'], m['split'], np.nan, np.nan, np.nan, np.nan, r['found'], dryr['found'])); continue
    t_true = np.exp(np.mean(np.log(tr['T60'][mid]))); t_est = np.exp(np.mean(np.log(r['T60'][mid])))
    d_true = np.mean(tr['drr_db'][mid]); d_est = np.mean(r['drr_db'][mid])
    d_dry = np.mean(dryr['drr_db'][mid]) if dryr['found'] else np.nan
    rows.append((m['effect'], m['split'], t_true, t_est, d_true, d_est, d_dry, dryr['T60'][mid].mean() if dryr['found'] else np.nan))
print('%-22s %-5s %7s %7s %7s | %7s %7s %7s | %s' % ('effect', 'split', 'T60true', 'T60est', 'ratio', 'DRRtrue', 'DRRest', 'err', 'dry source: DRR est / T60 est'))
for r in rows:
    print('%-22s %-5s %7.2f %7.2f %7.2f | %7.1f %7.1f %7.1f | %s' % (r[0], r[1], r[2], r[3], r[3] / r[2] if r[2] else np.nan, r[4], r[5], r[5] - r[4], '%5.1f / %.2f' % (r[6], r[7]) if isinstance(r[6], float) else str(r[6])))
a = np.array([[r[2], r[3], r[4], r[5]] for r in rows], float); ok = np.isfinite(a).all(1)
lr = np.log2(a[ok, 1] / a[ok, 0]); de = a[ok, 3] - a[ok, 2]
print('cases %d, usable %d | T60: median |log2 ratio| %.2f (x%.2f), within 25%%: %d%% | DRR error: median %.1f dB, median abs %.1f dB | %.0f s' % (len(rows), ok.sum(), np.median(np.abs(lr)), 2 ** np.median(np.abs(lr)), 100 * np.mean(np.abs(lr) < np.log2(1.25)), np.median(de), np.median(np.abs(de)), time.time() - t0))
