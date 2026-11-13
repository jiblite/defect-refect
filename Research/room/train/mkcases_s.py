# More training cases with WIDE sounds under the reverb. Over half of the dry clips are mono or nearly so, and the model learned
# "what left and right do not share is room": with a stereo sound under the reverb it takes parts of the sound for room (scores
# on the scoreboard: mean 8 against 24 with a mono sound underneath) and it finds a "room" on 60 % of dry stereo sounds.
# Here two or three clips sit at different places between left and right, so the dry sound itself has plenty that the channels
# do not share, with a reverb on the whole (or, on one case in four, none).    python3 mkcases_s.py first last [processes]
import sys, os, json, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, soundfile as sf, common as cm, engines as en, mkcases as mk
from multiprocessing import Pool
OUT = '/tmp/claude-0/v2/train/cases_s/'; os.makedirs(OUT, exist_ok=True); SR = 48000
def narrowed(x, w): m = x.mean(1); s = 0.5 * (x[:, 0] - x[:, 1]) * w; return np.stack([m + s, m - s], 1)
def placed(x, pos):
    """A clip at a place between left (0) and right (1): constant-power gains on its two channels (a stereo clip keeps its own picture, leaning)."""
    a = pos * np.pi / 2; return x * (np.sqrt(2.0) * np.array([np.cos(a), np.sin(a)]))[None]
def scene(rng):
    k = 2 if rng.random() < 0.65 else 3; names = [mk.CLIPS[int(rng.integers(len(mk.CLIPS)))] for _ in range(k)]; pos = [rng.uniform(0.0, 0.35), rng.uniform(0.65, 1.0)] + [rng.uniform(0.0, 1.0)] * (k - 2)
    if rng.random() < 0.5: pos[0], pos[1] = pos[1], pos[0]
    out = None; how = []
    for j, c in enumerate(names):
        x, sr = sf.read(mk.DRY + c + '.flac', always_2d=True); x, h = mk.treat(x.astype(np.float64), rng); g = 1.0 if j == 0 else 10 ** (rng.uniform(-9.0, 3.0) / 20)
        if out is None: out = np.zeros_like(x)
        n = min(len(out), len(x)); out[:n] += g * placed(x[:n], pos[j]); how.append('%s at %.2f, %.1f dB (%s)' % (c, pos[j], 20 * np.log10(g), h))
    pk = np.abs(out).max(); return out * (0.5 / max(pk, 1e-9)), names, how
def one(i):
    f = OUT + 's%05d.npz' % i
    if os.path.exists(f): return i, 'kept', 0.0
    rng = np.random.default_rng(550000 + i); t0 = time.time(); x, names, how = scene(rng); n = len(x) + 3 * SR; xp = np.vstack([x, np.zeros((3 * SR, 2))]); u = rng.random()
    info = dict(clip='+'.join(names), kind='scene of %d' % len(names), treat='; '.join(how))
    try:
        if u < 0.25: wet = xp.copy(); direct = xp.copy(); info['engine'] = 'none'
        elif u < 0.46:
            h = en.room_ir(int(rng.integers(len(en.ROOMS)))); wet = en.conv22(xp, h, n); direct = en.conv22(xp, en.ir_direct(h), n); info['engine'] = 'room'
        elif u < 0.62:
            h, si = en.stat_ir(rng); wet = en.conv22(xp, h, n); direct = en.conv22(xp, en.ir_direct(h), n); info['engine'] = 'stat'; info.update(si)
        else:
            name = en.PLUGINS[int(rng.integers(len(en.PLUGINS)))]; p = en.ENG[name][0](rng); wet = en.run_plugin(name, xp, p); taps, off = en.plugin_direct(name, p); direct = en.conv22(xp, taps, n + 400)
            direct = (np.vstack([np.zeros((off, 2)), direct]) if off >= 0 else direct[-off:])[:n]; info['engine'] = name
            if len(direct) < n: direct = np.vstack([direct, np.zeros((n - len(direct), 2))])
    except Exception as ex:
        return i, 'FAILED ' + str(ex)[:160], 0.0
    if not (np.isfinite(wet).all() and np.isfinite(direct).all()): return i, 'NAN', 0.0
    wd = 1.0 if rng.random() < 0.7 else float(10 ** rng.uniform(-1.0, -0.05)); wet, direct = narrowed(wet, wd), narrowed(direct, wd)
    ed = (direct ** 2).sum(); er = ((wet - direct) ** 2).sum(); drr = 10 * np.log10(max(ed, 1e-20) / max(er, 1e-20)); info['drr'] = float(drr); info['width'] = wd
    if ed < 1e-6: return i, 'no direct sound', 0.0
    if info['engine'] != 'none' and not (-14.0 < drr < 26.0): return i, 'skipped: DRR %.1f (%s)' % (drr, info['engine']), 0.0
    ft = cm.features(wet, direct); np.savez_compressed(f, info=json.dumps(info), **{k: v.astype(np.float16) for k, v in ft.items()})
    return i, info['engine'], time.time() - t0
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); NP = int(sys.argv[3]) if len(sys.argv) > 3 else 3; t0 = time.time(); import collections; cnt = collections.Counter()
    with Pool(NP) as p:
        for k, r in enumerate(p.imap_unordered(one, range(a, b), chunksize=2)):
            cnt[r[1].split(':')[0].split(' ')[0]] += 1
            if k % 100 == 0 or r[1].startswith('FAILED') or r[1] == 'NAN': print(k, r, '%.0f s' % (time.time() - t0), flush=True)
    print(dict(cnt), '%.0f s' % (time.time() - t0))
