# More training cases, for sounds with LITTLE difference between left and right. The first set (mkcases.py) only ever narrowed
# the dry clip, the reverb on top was always full stereo, and the model learned to lean on that: on a mono file it finds no room
# at all, and at a quarter of the width hardly any (narrow_test.py). Here the finished sound is narrowed AS A WHOLE (direct part
# and reverb alike, as a narrow or mono recording of a room would be), and every rendered case is stored twice: once at a
# random width and once in mono.     python3 mkcases_w.py first last [processes]      -> cases_w/c<i>n.npz and c<i>m.npz
import sys, os, json, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, soundfile as sf, common as cm, engines as en, mkcases as mk
from multiprocessing import Pool
OUT = '/tmp/claude-0/v2/train/cases_w/'; os.makedirs(OUT, exist_ok=True); SR = 48000
def narrowed(x, w): m = x.mean(1); s = 0.5 * (x[:, 0] - x[:, 1]) * w; return np.stack([m + s, m - s], 1)
def one(i):
    fn, fm = OUT + 'c%05dn.npz' % i, OUT + 'c%05dm.npz' % i
    if os.path.exists(fn) and os.path.exists(fm): return i, 'kept', 0.0
    rng = np.random.default_rng(770000 + i); t0 = time.time(); clip = mk.CLIPS[int(rng.integers(len(mk.CLIPS)))]
    x, sr = sf.read(mk.DRY + clip + '.flac', always_2d=True); x, how = mk.treat(x.astype(np.float64), rng); n = len(x) + 3 * SR; xp = np.vstack([x, np.zeros((3 * SR, 2))]); u = rng.random(); info = dict(clip=clip, kind=mk.dmeta[clip], treat=how)
    try:
        if u < 0.12: wet = xp.copy(); direct = xp.copy(); info['engine'] = 'none'
        elif u < 0.37:
            h = en.room_ir(int(rng.integers(len(en.ROOMS)))); wet = en.conv22(xp, h, n); direct = en.conv22(xp, en.ir_direct(h), n); info['engine'] = 'room'
        elif u < 0.55:
            h, si = en.stat_ir(rng); wet = en.conv22(xp, h, n); direct = en.conv22(xp, en.ir_direct(h), n); info['engine'] = 'stat'; info.update(si)
        else:
            name = en.PLUGINS[int(rng.integers(len(en.PLUGINS)))]; p = en.ENG[name][0](rng); wet = en.run_plugin(name, xp, p); taps, off = en.plugin_direct(name, p); direct = en.conv22(xp, taps, n + 400)
            direct = (np.vstack([np.zeros((off, 2)), direct]) if off >= 0 else direct[-off:])[:n]; info['engine'] = name
            if len(direct) < n: direct = np.vstack([direct, np.zeros((n - len(direct), 2))])
    except Exception as ex:
        return i, 'FAILED ' + str(ex)[:160], 0.0
    if not (np.isfinite(wet).all() and np.isfinite(direct).all()): return i, 'NAN', 0.0
    r2 = np.random.default_rng(990000 + i); w = float(10 ** r2.uniform(-1.3, -0.05)); made = 0
    for f, wd in ((fn, w), (fm, 0.0)):
        ww, dd = narrowed(wet, wd), narrowed(direct, wd); ed = (dd ** 2).sum(); er = ((ww - dd) ** 2).sum(); drr = 10 * np.log10(max(ed, 1e-20) / max(er, 1e-20))
        if ed < 1e-6 or (info['engine'] != 'none' and not (-14.0 < drr < 26.0)): continue
        ft = cm.features(ww, dd); np.savez_compressed(f, info=json.dumps(dict(info, drr=float(drr), width=wd)), **{k: v.astype(np.float16) for k, v in ft.items()}); made += 1
    return i, info['engine'] if made else 'skipped', time.time() - t0
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); NP = int(sys.argv[3]) if len(sys.argv) > 3 else 3; t0 = time.time(); import collections; cnt = collections.Counter()
    with Pool(NP) as p:
        for k, r in enumerate(p.imap_unordered(one, range(a, b), chunksize=2)):
            cnt[r[1].split(':')[0].split(' ')[0]] += 1
            if k % 100 == 0 or r[1].startswith('FAILED') or r[1] == 'NAN': print(k, r, '%.0f s' % (time.time() - t0), flush=True)
    print(dict(cnt), '%.0f s' % (time.time() - t0))
