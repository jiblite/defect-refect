# Room capture scored end to end: measure the room on sound A, put it on dry sound B, compare with the real B through the real reverb.
import sys, os, json, glob, time, pickle
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, score, gbfeat, roomcap as rc
from multiprocessing import Pool
corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); MODE = os.environ.get('GAINS', 'true'); MODEL = None
def gains_for(ft):
    global MODEL
    if MODE == 'true': return gbfeat.target(ft['LE'], ft['LS'])[0]
    import predict; return predict.gains(ft)
def one(d):
    cA = truth.load(d, 'A'); dA, rA = truth.parts(cA); ft = cm.features(cA['wet'], dA); g = gains_for(ft); S, R, va, bo = rc.tracks(ft['LE'], g); W = rc.energy_response(S, R, valid=va, band_ok=bo); nb = rc.numbers(W)
    # how wide the reverb part is: unshared share of the wet sound where the gains say "almost all room"
    U = 10 ** (np.clip(ft['U256'], -40, 0) / 10); tail = (g < -12.0) & (ft['LE'] > ft['LE'].max() - 50.0)
    u = np.array([np.median(U[tail[:, b], b]) if tail[:, b].sum() >= 10 else np.nan for b in range(cm.NB)]); u = np.where(np.isfinite(u), u, np.nanmedian(u) if np.isfinite(u).any() else 1.0)
    h = rc.synth_ir(W, np.clip(u * 1.15, 0.0, 1.0), seed=3)
    cB = truth.load(d, 'B'); sc0 = score.Scorer(cB['wet']); base = sc0(cB['dry']); out = {}
    def sc(y): return 100.0 * (1.0 - sc0(y) / max(base, 1e-9))
    for s in (0.5, 1.0, 2.0): out['amount %.1f' % s] = sc(rc.apply_room(cB['dry'], h, s))
    if os.environ.get('NUMBERS'):
        # the room as numbers (strength + decay time per band) read with the mask's help, as a plain exponential decay per band
        import roomnum, param_ceiling as pc
        rn = roomnum.room_numbers(ft['LE'], g); kt_ = (10 ** (ft['LR'] / 10)).sum(0) / np.maximum((10 ** (ft['LS'] / 10)).sum(0), 1e-30); m_ = (cm.FC41 > 450) & (cm.FC41 < 7000)
        if rn is None: out['numbers'] = 0.0; out['numbers +2 dB'] = 0.0; out['_num kerr'] = np.nan; out['_num T60'] = np.nan
        else:
            kdb, T60n, evn = rn; out['numbers'] = sc(rc.apply_room(cB['dry'], rc.synth_ir(pc.expo(10 ** (kdb / 10), T60n, 0), np.clip(u * 1.15, 0.0, 1.0), seed=3), 1.0)); out['numbers +2 dB'] = sc(rc.apply_room(cB['dry'], rc.synth_ir(pc.expo(10 ** ((kdb + 2.0) / 10), T60n, 0), np.clip(u * 1.15, 0.0, 1.0), seed=3), 1.0))
            out['_num kerr'] = float(np.median(kdb[m_] - 10 * np.log10(kt_[m_] + 1e-9))); out['_num T60'] = float(np.median(T60n[m_]))
    if os.environ.get('ORACLE'):
        # ceilings: (1) the room read from the true parts of A; (2) scored on the reverb part alone (B's true direct part given, so that what the effect did to the direct sound does not count)
        W0 = rc.energy_response(10 ** (ft['LS'] / 10), 10 ** (ft['LR'] / 10), c1=0.0, tie=0, iters=4); h0 = rc.synth_ir(W0, np.clip(u * 1.15, 0.0, 1.0), seed=3); out['room from true parts'] = sc(rc.apply_room(cB['dry'], h0, 1.0))
        dB_, rB_ = truth.parts(cB); b2 = sc0(dB_); out['reverb part only'] = 100.0 * (1.0 - sc0(dB_ + rc.apply_room(cB['dry'], h, 1.0) - cB['dry']) / max(b2, 1e-9)); out['reverb part only, true parts'] = 100.0 * (1.0 - sc0(dB_ + rc.apply_room(cB['dry'], h0, 1.0) - cB['dry']) / max(b2, 1e-9))
        out['direct part alone'] = sc(dB_)
        # the room's energy response read straight off the click responses (both inputs together), at the engine's own detail: what the synthesis can do when the measurement is perfect
        a_, b_, pk_ = truth.direct_window(cA['h']); hs = cA['h'][0] + cA['h'][1]; nn = rc.K * cm.HOP + 2 * cm.NF; dd = np.zeros((nn, 2)); dd[:b_ - a_] = hs[a_:b_]; tt = hs[:nn].copy(); tt[:b_] = 0.0; tt = np.vstack([tt, np.zeros((max(0, nn - len(tt)), 2))])
        if a_ > 0: tt = tt[a_:]; tt = np.vstack([tt, np.zeros((nn - len(tt), 2))])
        Pd = cm.bandpow(cm.fwd(dd)); Pt = cm.bandpow(cm.fwd(tt)); c0 = int(np.argmax(Pd.sum(1))); Wc = (Pt[c0:c0 + rc.K] / np.maximum(Pd.sum(0), 1e-30)[None]).T; Wc = np.pad(Wc, ((0, 0), (0, rc.K - Wc.shape[1])))
        hc = rc.synth_ir(Wc, np.clip(u * 1.15, 0.0, 1.0), seed=3); out['room from the click'] = sc(rc.apply_room(cB['dry'], hc, 1.0)); out['_kappa click'] = float(10 * np.log10(Wc.sum(1)[(cm.FC41 > 450) & (cm.FC41 < 7000)].mean() + 1e-9)); out['_kappa fit'] = float(10 * np.log10(W0.sum(1)[(cm.FC41 > 450) & (cm.FC41 < 7000)].mean() + 1e-9))
        # a statistical twin of the true room: the true tails with their fine structure scrambled (same time and frequency energy at 5 ms / 190 Hz and 21 ms / 47 Hz detail, same left/right relation)
        import scipy.signal as ss
        a_, b_, pk_ = truth.direct_window(cB['h']); n = len(cB['wet'])
        for nm, nseg in (('twin 21 ms', 1024), ('twin 85 ms', 4096)):
            y = dB_.copy(); rng = np.random.default_rng(7)
            for i in range(2):
                tl = cB['h'][i].copy(); tl[:b_] = 0.0; f_, t_, Z = ss.stft(tl.T, nperseg=nseg, noverlap=nseg * 3 // 4); Z = Z * np.exp(2j * np.pi * rng.random(Z.shape[1:]))[None]; tw = ss.istft(Z, nperseg=nseg, noverlap=nseg * 3 // 4)[1].T
                for o in range(2): y[:, o] += ss.fftconvolve(cB['dry'][:, i], tw[:, o])[:n]
            out[nm] = sc(y)
    t = truth.room_truth(cA); mid41 = (cm.FC41 > 450) & (cm.FC41 < 7000); mid20 = (np.asarray(__import__('room').FC) > 450) & (np.asarray(__import__('room').FC) < 7000)
    out['_T60 found'] = float(np.nanmedian(nb['T60'][mid41])); out['_T60 true'] = float(np.nanmedian(13.8155 * t['tau'][mid20])); out['_DRR found'] = float(-10 * np.log10(np.mean(nb['kappa'][mid41]) + 1e-9)); out['_DRR true'] = float(-10 * np.log10(np.nanmean(t['kappa'][mid20]) + 1e-9))
    m = cA['meta']; return cA['name'], m['split'], corp[m['A']]['kind'], out['_DRR true'], base, out
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '2'))) as p: rows = p.map(one, ds)
    keys = [k for k in rows[0][5].keys() if not k.startswith('_')]; a = np.array([[r[3], r[4]] + [r[5][k] for k in keys] for r in rows]); sp = np.array([r[1] for r in rows])
    if '-v' in sys.argv:
        for r in rows: print('%-24s %-5s %-16s DRR %5.1f (found %5.1f)  T60 %.2f (found %.2f) | B dry is %4.1f dB from B wet | ' % (r[0], r[1], r[2], r[3], r[5]['_DRR found'], r[5]['_T60 true'], r[5]['_T60 found'], r[4]) + ' | '.join('%s %6.1f' % (k, r[5][k]) for k in keys))
    tr = np.array([np.log2(r[5]['_T60 found'] / r[5]['_T60 true']) for r in rows]); tr = tr[np.isfinite(tr)]; de = np.array([r[5]['_DRR found'] - r[5]['_DRR true'] for r in rows])
    print('gains: %s | decay time: median x%.2f off, within 25%% in %.0f%% | room loudness: median error %.1f dB, median |error| %.1f dB' % (MODE, 2 ** np.median(np.abs(tr)), 100 * np.mean(np.abs(tr) < np.log2(1.25)), np.median(de), np.median(np.abs(de))))
    if '_num kerr' in rows[0][5]:
        ke = np.array([r[5]['_num kerr'] for r in rows]); tn = np.array([np.log2(r[5]['_num T60'] / r[5]['_T60 true']) for r in rows]); f = np.isfinite(ke) & np.isfinite(tn)
        print('numbers: answered %.0f%% | strength: median error %+.1f dB, median |error| %.1f dB | decay time: median x%.2f off, within 20%% in %.0f%%' % (100 * f.mean(), np.median(ke[f]), np.median(np.abs(ke[f])), 2 ** np.median(np.abs(tn[f])), 100 * np.mean(np.abs(tn[f]) < np.log2(1.2))))
    for lab, sel in [('all', np.ones(len(a), bool)), ('reverbs also used in training', sp == 'dev'), ('reverbs and sounds never seen', sp == 'test'), ('light (DRR >= 8)', a[:, 0] >= 8), ('middle', (a[:, 0] < 8) & (a[:, 0] >= 0)), ('wet (DRR < 0)', a[:, 0] < 0)]:
        if sel.sum(): print('%-32s n=%3d | ' % (lab, sel.sum()) + ' | '.join('%s: median %5.1f, below zero %2.0f%%' % (k, np.median(a[sel, 2 + i]), 100 * np.mean(a[sel, 2 + i] < 0)) for i, k in enumerate(keys)))
    print('%.0f s' % (time.time() - t0))
