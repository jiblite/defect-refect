# Training cases for what the room model had not seen before 0.4.5:
#   - sounds with one of the OTHER effects on them (flanger, chorus, phaser, stereo delay, widening, echoes, distortion,
#     bitcrush, filter sweep, tremolo, auto pan, saturation, exciter, ring modulation, filter, stutter, gate), with a reverb
#     after the effect or without one. The effect belongs to the sound: the direct part is the sound WITH its effect.
#     Only the 'dev' tools of the effect zoo are used; the 'test' tools stay unseen for the checks.
#   - sounds that are CUT OFF while still playing: every rendered case is stored twice, the whole sound with 3 s after it
#     (as in the other sets) and the same sound cut off at 6 s, with no silence and no tail after it; and a third time as
#     an EXCERPT, 4.5 s from somewhere inside it: cut at both ends, with the reverb of what came before ringing into it.
# The model had learned from sounds that end inside the file, where a room shows as the tail after the last note; on a
# file that simply stops it read a room into many dry sounds (Research/room/bench/cut_test.py), and it had never been
# shown a chorus or a stereo delay and told "this is not a room".
#   python3 mkcases_x.py first last [processes]     -> cases_x/x<i>.npz (849 frames), cases_xc/x<i>.npz (568 frames), cases_xm/x<i>.npz (427 frames)
import sys, os, json, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/detect/py')
import numpy as np, soundfile as sf, common as cm, engines as en, mkcases as mk, mkcases_s as mks
import zoo_fx as Z
from multiprocessing import Pool
OUT = '/tmp/claude-0/v2/train/cases_x/'; OUTC = '/tmp/claude-0/v2/train/cases_xc/'; OUTM = '/tmp/claude-0/v2/train/cases_xm/'; os.makedirs(OUT, exist_ok=True); os.makedirs(OUTC, exist_ok=True); os.makedirs(OUTM, exist_ok=True); SR = 48000
TOOLS = sorted(n for n, v in Z.ZOO.items() if v[1] == 'dev'); P_FX = 0.55; P_REV = 0.55; CUT = 6 * SR; MID = int(4.5 * SR)
def narrowed(x, w): m = x.mean(1); s = 0.5 * (x[:, 0] - x[:, 1]) * w; return np.stack([m + s, m - s], 1)
def one(i):
    f = OUT + 'x%05d.npz' % i; fc = OUTC + 'x%05d.npz' % i; fm = OUTM + 'x%05d.npz' % i
    if os.path.exists(f) and os.path.exists(fc) and os.path.exists(fm): return i, 'kept', 0.0
    rng = np.random.default_rng(330000 + i); t0 = time.time()
    if rng.random() < 0.65:
        clip = mk.CLIPS[int(rng.integers(len(mk.CLIPS)))]; x, sr = sf.read(mk.DRY + clip + '.flac', always_2d=True); x, how = mk.treat(x.astype(np.float64), rng); info = dict(clip=clip, kind=mk.dmeta[clip], treat=how)
    else:
        x, names, how = mks.scene(rng); info = dict(clip='+'.join(names), kind='scene of %d' % len(names), treat='; '.join(how))
    n = len(x) + 3 * SR; xp = np.vstack([x, np.zeros((3 * SR, 2))]); info['fx'] = 'none'
    try:
        if rng.random() < P_FX:
            for _ in range(4):
                name = TOOLS[int(rng.integers(len(TOOLS)))]; fam, split, samp, fn, clipf = Z.ZOO[name]
                if clipf is not None and not clipf(x): continue
                p = samp(rng); y = Z.apply(name, xp, p)
                if not np.isfinite(y).all() or np.abs(y[:CUT]).max() < 1e-3: continue
                xp = y * (0.5 / np.abs(y).max()); info['fx'] = name; info['fx_family'] = fam; info['fx_params'] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in p.items()}; break
        u = rng.random()
        if u >= P_REV: wet = xp.copy(); direct = xp.copy(); info['engine'] = 'none'
        elif u < 0.36 * P_REV:
            h = en.room_ir(int(rng.integers(len(en.ROOMS)))); wet = en.conv22(xp, h, n); direct = en.conv22(xp, en.ir_direct(h), n); info['engine'] = 'room'
        elif u < 0.62 * P_REV:
            h, si = en.stat_ir(rng); wet = en.conv22(xp, h, n); direct = en.conv22(xp, en.ir_direct(h), n); info['engine'] = 'stat'; info.update(si)
        else:
            name = en.PLUGINS[int(rng.integers(len(en.PLUGINS)))]; p = en.ENG[name][0](rng); wet = en.run_plugin(name, xp, p); taps, off = en.plugin_direct(name, p); direct = en.conv22(xp, taps, n + 400)
            direct = (np.vstack([np.zeros((off, 2)), direct]) if off >= 0 else direct[-off:])[:n]; info['engine'] = name
            if len(direct) < n: direct = np.vstack([direct, np.zeros((n - len(direct), 2))])
    except Exception as ex:
        return i, 'FAILED ' + str(ex)[:160], 0.0
    if not (np.isfinite(wet).all() and np.isfinite(direct).all()): return i, 'NAN', 0.0
    v = rng.random(); wd = 1.0 if v < 0.7 else 0.0 if v < 0.78 else float(10 ** rng.uniform(-1.0, -0.05)); wet, direct = narrowed(wet, wd), narrowed(direct, wd)
    ed = (direct ** 2).sum(); er = ((wet - direct) ** 2).sum(); drr = 10 * np.log10(max(ed, 1e-20) / max(er, 1e-20)); info['drr'] = float(drr); info['width'] = wd
    if ed < 1e-6: return i, 'no direct sound', 0.0
    if info['engine'] != 'none' and not (-14.0 < drr < 26.0): return i, 'skipped: DRR %.1f (%s)' % (drr, info['engine']), 0.0
    ft = cm.features(wet, direct); np.savez_compressed(f, info=json.dumps(dict(info, cut=0)), **{k: v.astype(np.float16) for k, v in ft.items()})
    ft = cm.features(wet[:CUT], direct[:CUT]); np.savez_compressed(fc, info=json.dumps(dict(info, cut=1)), **{k: v.astype(np.float16) for k, v in ft.items()})
    a = int(rng.uniform(0.5, 1.5) * SR)        # (drawn last, so the first two versions are the same with or without it)
    if (direct[a:a + MID] ** 2).sum() > 1e-6:
        ft = cm.features(wet[a:a + MID], direct[a:a + MID]); np.savez_compressed(fm, info=json.dumps(dict(info, cut=2, start=a / SR)), **{k: v.astype(np.float16) for k, v in ft.items()})
    else: np.savez_compressed(fm, info=json.dumps(dict(info, cut=2, start=a / SR, empty=1)))
    return i, info['fx'].split('_')[0] + ('+rev' if info['engine'] != 'none' else ''), time.time() - t0
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); NP = int(sys.argv[3]) if len(sys.argv) > 3 else 3; t0 = time.time(); import collections; cnt = collections.Counter()
    with Pool(NP) as p:
        for k, r in enumerate(p.imap_unordered(one, range(a, b), chunksize=2)):
            cnt[r[1].split(':')[0].split(' ')[0]] += 1
            if k % 200 == 0 or r[1].startswith('FAILED') or r[1] == 'NAN': print(k, r, '%.0f s' % (time.time() - t0), flush=True)
    print(dict(cnt), '%.0f s' % (time.time() - t0))
