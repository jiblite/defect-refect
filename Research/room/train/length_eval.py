# Does the Length fader on ONE sound do what it says? The truth is made from the real click responses: the reverb part is
# faded with an extra decay (so the mid bands end up f times as long), the direct part stays. Scored like everything else:
# 100 = exactly that, 0 = no better than leaving the sound alone. Two ways are compared: reshaping the sound's own reverb
# (engine.one_sound) and the earlier way (reverb out, found room rebuilt with the new length put back).
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, scipy.signal as ss, common as cm, truth, score, engine, render, rc_look, param_ceiling as pc, roomnum, roomcap as rc
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_b9'); ORACLE = os.environ.get('ORACLE', '') == '1'; FS = (0.5, 0.7, 1.5)
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; n = len(wet); ft = cm.features(wet, direct); gt = np.clip(ft['LS'] - ft['LE'], -30.0, 0.0); an = engine.analyse(wet, MODEL, g=gt if ORACLE else None)
    Tc = np.array([pc.t60_of(x) for x in rc_look.click_W(c)]); ok = roomnum.MID & np.isfinite(Tc); r = dict(name=c['name'], found=bool(an['found']))
    if not an['found'] or not ok.any(): return r
    Tm = float(np.median(Tc[ok])); a, b, pk = truth.direct_window(c['h']); r['t60'] = Tm; r['sc'] = {}
    for f in FS:
        delta = 60.0 / (f * Tm) - 60.0 / Tm; t = np.maximum(np.arange(len(c['h'][0])) - b, 0) / truth.SR; win = 10 ** (-delta * t / 20.0); hw = [hh * win[:, None] for hh in c['h']]
        for hh in hw: hh[:b] = 0.0
        tail = np.stack([ss.fftconvolve(c['dry'][:, 0], hw[0][:, o])[:n] + ss.fftconvolve(c['dry'][:, 1], hw[1][:, o])[:n] for o in range(2)], 1); target = direct + tail; sc = score.Scorer(target); base = sc(wet)
        T1 = 1.0 / (1.0 / an['T60'] + delta / 60.0); T1 = np.where(T1 > 0, T1, 20.0); T1 = np.clip(T1, 0.03, 20.0)
        new = render.reshaped(wet, an['g'], 1.0, engine.reshape_factor(an, T1))
        an2 = dict(an); r0 = 10 ** (-6.0 * cm.DT / an['T60']); r1 = 10 ** (-6.0 * cm.DT / T1); an2['T60'] = T1; an2['kdb'] = an['kdb'] + 10 * np.log10((1 - r0) / (1 - r1)); dd = render.render(wet, an['g'], 0.0); old = dd + (rc.apply_room(dd, engine.room_ir(an2), 1.0) - dd)
        r['sc'][str(f)] = dict(new=float(100 * (1 - sc(new) / max(base, 1e-9))), old=float(100 * (1 - sc(old) / max(base, 1e-9))), base=float(base))
    return r
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '2'))) as p: rows = p.map(one, ds)
    tag = 'oracle' if ORACLE else MODEL; json.dump(rows, open('/tmp/claude-0/v2/train/length_eval_%s.json' % tag, 'w')); ok = [r for r in rows if 'sc' in r]; print('mask: %s | %d cases' % (tag, len(ok)))
    for f in FS:
        for way, lab in (('new', "reshaping the sound's own reverb"), ('old', 'rebuilt room put back')):
            v = np.array([r['sc'][str(f)][way] for r in ok]); bs = np.array([r['sc'][str(f)]['base'] for r in ok]); pooled = 100 * (1 - (bs * (1 - v / 100)).sum() / bs.sum())
            print('length x%.1f | %-34s median %6.1f mean %6.1f below zero %3.0f%% pooled %6.1f' % (f, lab, np.median(v), np.mean(v), 100 * np.mean(v < 0), pooled))
    print('%.0f s' % (time.time() - t0))
