# Training cases: a dry clip (with a random stereo treatment), a reverb, and the exact direct part; stored as what the model
# sees (band levels, unshared share) and what it has to find (direct and reverb band levels).
import sys, os, json, time, hashlib
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, soundfile as sf, scipy.signal as ss, common as cm, engines as en
from multiprocessing import Pool
DRY = '/tmp/claude-0/v2/train/dry/'; OUT = '/tmp/claude-0/v2/train/cases/'; os.makedirs(OUT, exist_ok=True); SR = 48000
dmeta = json.load(open('/tmp/claude-0/v2/train/dry_meta.json')); CLIPS = sorted(dmeta.keys())
def treat(x, rng):
    """A random but plausible stereo picture and tone for the dry clip."""
    m = rng.random(); how = 'as is'
    if m < 0.12: x = np.repeat(x.mean(1, keepdims=True), 2, 1); how = 'mono'
    elif m < 0.27:
        w = rng.uniform(0.2, 1.6); mid = x.mean(1); side = 0.5 * (x[:, 0] - x[:, 1]) * w; x = np.stack([mid + side, mid - side], 1); how = 'width %.2f' % w
    elif m < 0.37:
        d = rng.uniform(-12, 12); x = x * np.array([10 ** (d / 40), 10 ** (-d / 40)])[None]; how = 'balance %.1f dB' % d
    elif m < 0.43:
        d = rng.uniform(0.05, 0.8) * 1e-3 * SR; c = int(rng.integers(0, 2)); X = np.fft.rfft(x[:, c]); x = x.copy(); x[:, c] = np.fft.irfft(X * np.exp(-2j * np.pi * np.fft.rfftfreq(len(x)) * d), len(x)); how = 'delay %.2f ms' % (d / SR * 1e3)
    if rng.random() < 0.5:
        a = rng.uniform(-0.2, 0.2); X = np.fft.rfft(x, axis=0); f = np.maximum(np.fft.rfftfreq(len(x), 1 / SR), 30.0); x = np.fft.irfft(X * ((f / 1000.0) ** a)[:, None], len(x), axis=0); how += ', tilt %.2f' % a
    pk = np.abs(x).max(); return x * (0.5 / max(pk, 1e-9)), how
def one(i):
    f = OUT + 'c%05d.npz' % i
    if os.path.exists(f): return i, 'kept', 0.0
    rng = np.random.default_rng(770000 + i); t0 = time.time(); clip = CLIPS[int(rng.integers(len(CLIPS)))]
    x, sr = sf.read(DRY + clip + '.flac', always_2d=True); x, how = treat(x.astype(np.float64), rng); n = len(x) + 3 * SR; xp = np.vstack([x, np.zeros((3 * SR, 2))]); u = rng.random(); info = dict(clip=clip, kind=dmeta[clip], treat=how)
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
    ed = (direct ** 2).sum(); er = ((wet - direct) ** 2).sum(); drr = 10 * np.log10(max(ed, 1e-20) / max(er, 1e-20)); info['drr'] = float(drr)
    if ed < 1e-6: return i, 'no direct sound', 0.0
    if info['engine'] != 'none' and not (-14.0 < drr < 26.0): return i, 'skipped: DRR %.1f (%s)' % (drr, info['engine']), 0.0
    ft = cm.features(wet, direct); np.savez_compressed(f, info=json.dumps(info), **{k: v.astype(np.float16) for k, v in ft.items()})
    return i, info['engine'], time.time() - t0
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time(); import collections; cnt = collections.Counter()
    with Pool(4) as p:
        for k, r in enumerate(p.imap_unordered(one, range(a, b), chunksize=2)):
            cnt[r[1].split(':')[0].split(' ')[0]] += 1
            if k % 100 == 0 or r[1].startswith('FAILED') or r[1] == 'NAN': print(k, r, '%.0f s' % (time.time() - t0), flush=True)
    print(dict(cnt), '%.0f s' % (time.time() - t0))
