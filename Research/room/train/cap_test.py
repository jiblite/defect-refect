# Can the found room hold the answer back where it goes too deep? A room with a given decay and strength cannot put more
# into a cell than the sound before it, smeared with that decay, times the strength. The answer is capped so that it
# never takes out more than that bound plus a margin: g' = max(g, 10 log10(1 - bound * margin / level)).
# Bounds:  app   the app's own numbers and smear (roomnum, from the answer)
#          wet   the wet sound itself smeared (no trust in the answer under the sound), strength fitted on the cells the
#                answer calls all room (the free decays)
#          rec   the split run forward in time: room(t+1) = r * (room(t) + k * direct(t)), direct = level - room, k fitted
#                so that the room accounts for the free decays
#          true  the true reverb level (what a perfect bound would give)
# Scored in the band picture: whole file, the sound itself (first 6 s), the ring-out; scoreboard reverbs, both sounds.
#   NP=4 python3 cap_test.py   -> cap_test.json
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, roomnum, body_diag as bd
from multiprocessing import Pool
T = '/tmp/claude-0/v2/train/'; MODEL = os.environ.get('MODEL', 'unet_x+unet_xp2_e5'); MARG = (0.0, 3.0, 6.0, 9.0)
def smear_of(S, T60):
    P = np.zeros_like(S)
    for b in range(S.shape[1]):
        r = 10 ** (-6.0 * cm.DT / T60[b]) if np.isfinite(T60[b]) else 0.0; acc = 0.0
        for t in range(S.shape[0]): acc = r * acc + (1.0 - r) * S[t, b]; P[t, b] = acc
    return P
def rec_room(E, r, k):
    """One band: room and direct run forward. room(t+1) = r * (room(t) + k * direct(t)); direct = max(level - room, 0)."""
    R = np.zeros(len(E)); acc = 0.0
    for t in range(len(E)): R[t] = acc; acc = r * (acc + k * max(E[t] - acc, 0.0))
    return R
def cap(g, ratio_db, m): return np.maximum(g, 10 * np.log10(np.clip(1.0 - 10 ** ((ratio_db + m) / 10.0), 1e-3, 1.0)))
def one(f):
    import predict
    z = np.load(f); ft = {k: z[k].astype(np.float64) for k in ('LE', 'U64', 'U256')}; LE = ft['LE']; LS = z['LS'].astype(np.float64); LR = z['LR'].astype(np.float64); m = json.loads(str(z['info'])); g = predict.gains(ft, MODEL)
    r = dict(split=m['split'], drr=float(m['drr']), sc={}); r['sc']['answer'] = [p[0] or 0.0 for p in bd.parts(g, LE, LS)]; E = 10 ** (LE / 10.0); fl = LE.max() - 57.0
    for mg in MARG: r['sc']['true +%d' % mg] = [p[0] or 0.0 for p in bd.parts(cap(g, LR - LE, mg), LE, LS)]
    rn = roomnum.room_numbers(LE, g)
    if rn is None:
        for nm in ('app', 'wet', 'rec'):
            for mg in MARG: r['sc']['%s +%d' % (nm, mg)] = r['sc']['answer']
        r['found'] = False; return r
    kdb, T60, ev = rn; r['found'] = True; P = roomnum.smear(LE, g, T60); ra = 10 * np.log10(10 ** ((kdb + roomnum.BUILD_UP_DB) / 10.0)[None, :] * P / np.maximum(E, 1e-30) + 1e-9)
    PE = smear_of(np.where(LE > fl, E, 0.0), T60); allroom = (g < -10.0) & (LE > fl); rw = np.full(LE.shape, -99.0); rr = np.full(LE.shape, -99.0); kw = np.full(cm.NB, np.nan); kr = np.full(cm.NB, np.nan)
    for b in range(cm.NB):
        rows = allroom[:, b] & (PE[:, b] > 1e-4 * PE[:, b].max()) if PE[:, b].max() > 0 else np.zeros(len(LE), bool)
        if rows.sum() < 8: continue
        q = np.sort(np.log(E[rows, b] / PE[rows, b])); k = int(len(q) * 0.1); q = q[k:len(q) - k] if len(q) - 2 * k >= 4 else q; kw[b] = np.exp(np.mean(q))
        rdec = 10 ** (-6.0 * cm.DT / T60[b]); Eb = np.where(LE[:, b] > fl, E[:, b], 0.0); lo, hi = -9.0, 3.0          # log10 of k: the room must account for the free decays (the median of level / room there is 1)
        for _ in range(18):
            mid = 0.5 * (lo + hi); R = rec_room(Eb, rdec, 10 ** mid); med = np.median(np.log(np.maximum(R[rows], 1e-30) / E[rows, b]))
            if med < 0: lo = mid
            else: hi = mid
        kr[b] = 10 ** (0.5 * (lo + hi))
    okw = np.isfinite(kw); okr = np.isfinite(kr)
    if okw.any():
        kws = roomnum.smooth_fill(kw, okw.astype(float)); rw = 10 * np.log10(kws[None, :] * PE / np.maximum(E, 1e-30) + 1e-9)
    if okr.any():
        krs = roomnum.smooth_fill(kr, okr.astype(float))
        for b in range(cm.NB): rr[:, b] = 10 * np.log10(rec_room(np.where(LE[:, b] > fl, E[:, b], 0.0), 10 ** (-6.0 * cm.DT / T60[b]), krs[b]) / np.maximum(E[:, b], 1e-30) + 1e-9)
    for nm, ratio in (('app', ra), ('wet', rw), ('rec', rr)):
        for mg in MARG: r['sc']['%s +%d' % (nm, mg)] = [p[0] or 0.0 for p in bd.parts(cap(g, ratio, mg) if ratio.max() > -98 else g, LE, LS)]
    live = (LE[:562] > fl); tr = (LR - LE)[:562]                                                           # how far each bound is from the true reverb level, under the sound
    r['off'] = {nm: [float(np.median((ratio[:562] - tr)[live])), float(np.percentile((ratio[:562] - tr)[live], 10)), float(np.percentile((ratio[:562] - tr)[live], 90))] for nm, ratio in (('app', ra), ('wet', rw), ('rec', rr))}
    return r
def show(rows):
    drr = np.array([r['drr'] for r in rows]); names = list(rows[0]['sc'])
    for part, pn in enumerate(('whole file', 'the sound itself', 'the ring-out')):
        print(pn); print('  %-10s | %-30s | %-30s | %-30s' % ('', 'all (n=%d)' % len(rows), 'heavy (n=%d)' % (drr < 5).sum(), 'light, 5 dB and more (n=%d)' % (drr >= 5).sum()))
        for nm in names:
            S = np.array([r['sc'][nm][part] for r in rows]); print('  %-10s | %s' % (nm, ' | '.join('med %5.1f mean %6.1f worse %3.0f%%' % (np.median(S[s]), S[s].mean(), 100 * (S[s] < 0).mean()) for s in (np.ones(len(rows), bool), drr < 5, drr >= 5))))
    print('room found on %d of %d' % (sum(r['found'] for r in rows), len(rows)))
    for nm in ('app', 'wet', 'rec'):
        o = np.array([r['off'][nm] for r in rows if r.get('off')]); print('bound "%s" against the true reverb level under the sound, dB: median of the medians %.1f (10th %.1f, 90th %.1f); within a sound 10th to 90th: %.1f to %.1f' % (nm, np.median(o[:, 0]), np.percentile(o[:, 0], 10), np.percentile(o[:, 0], 90), np.median(o[:, 1]), np.median(o[:, 2])))
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--show': show(json.load(open(T + 'cap_test.json'))); sys.exit(0)
    t0 = time.time(); fs = sorted(glob.glob(T + 'sb/*.npz'))
    with Pool(int(os.environ.get('NP', '4'))) as p: rows = p.map(one, fs, chunksize=2)
    json.dump(rows, open(T + 'cap_test.json', 'w')); show(rows); print('%.0f s' % (time.time() - t0))
