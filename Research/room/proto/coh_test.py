import sys, json, glob, os
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, soundfile as sf, room, collections
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/reverb/'; C = '/tmp/claude-0/v2/corpus/dry/'; corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
def rd(p): return sf.read(p, always_2d=True)[0].astype(np.float64)
M = room.band_matrix(2048)
def coh(x):
    L = room.stft(x[:, 0], 2048, 512); R = room.stft(x[:, 1], 2048, 512)
    cross = np.abs((L * np.conj(R)).sum(0)); pl = (np.abs(L) ** 2).sum(0); pr = (np.abs(R) ** 2).sum(0)      # per bin, summed over the whole file
    return (M @ cross) / np.sqrt((M @ pl) * (M @ pr) + 1e-30)
def one(d):
    m = json.load(open(d + 'meta.json')); name = os.path.basename(d.rstrip('/')); z = np.load('/tmp/claude-0/v2/proto/cache/reverb/' + name + '.npz')
    tt, tA = z['true_tau'], z['true_A']; mid = (room.FC > 450) & (room.FC < 7000) & np.isfinite(tt) & np.isfinite(tA)
    if mid.sum() < 3: return None
    cw = coh(rd(d + 'A_wet.flac')); cd = coh(rd(C + m['A'] + '.wav')); ci = coh(rd(d + 'ir.flac')[480 + 1440:])      # IR tail only (30 ms after the click): how decorrelated is this reverb's tail?
    return (name, corp[m['A']]['kind'], float(np.mean(-10 * np.log10(tA[mid] * tt[mid]))), float(np.mean(cw[mid])), float(np.mean(cd[mid])), float(np.mean(ci[mid])))
if __name__ == '__main__':
    with Pool(4) as p: rows = [r for r in p.map(one, sorted(glob.glob(S + '*/'))) if r]
    for r in rows: print('%-24s %-16s DRR true %6.1f | coherence wet %.2f  dry %.2f  IR tail %.2f | DRR from coherence %6.1f (err %5.1f)' % (r[0], r[1], r[2], r[3], r[4], r[5], -10 * np.log10(max(1 - r[3], 1e-3) / max(r[3], 1e-3)), -10 * np.log10(max(1 - r[3], 1e-3) / max(r[3], 1e-3)) - r[2]))
    a = np.array([[r[2], -10 * np.log10(max(1 - r[3], 1e-3) / max(r[3], 1e-3)), r[4], r[5]] for r in rows])
    for lab, sel in [('all', np.ones(len(a), bool)), ('dry source coherent (>0.95) and IR tail decorrelated (<0.3)', (a[:, 2] > 0.95) & (a[:, 3] < 0.3)), ('dry source coherent (>0.95)', a[:, 2] > 0.95), ('dry source not coherent', a[:, 2] <= 0.95)]:
        e = a[sel, 1] - a[sel, 0]
        if len(e) > 2: print('%-62s n=%3d  corr %.2f  slope %.2f  median err %5.1f  median |err| %4.1f  |err|<4 dB: %3.0f%%' % (lab, len(e), np.corrcoef(a[sel, 0], a[sel, 1])[0, 1], np.polyfit(a[sel, 0], a[sel, 1], 1)[0], np.median(e), np.median(np.abs(e)), 100 * np.mean(np.abs(e) < 4)))
