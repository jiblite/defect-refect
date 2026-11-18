# Fit data for the moving-tone model: training clips through moving filters (this study's own and, for the 'dev' half of the
# outside tools, the scoreboard's kind of phaser and filter sweep with new settings), and sounds that have none: plain,
# through a filter that stands still, through a level movement, through the other effect families' outside tools.
# Per example: the wet picture, the true tone curve per frame and band (read with the probe signal through the same effect),
# which cells the dry sound fills, and per frame what is on (0 nothing, 1 filter sweep, 2 phaser).
#   python3 mkdata.py out_dir n_examples [seed] [jobs]          PHASER=0.65: the share of phasers among the moving filters (0.45)
import sys, os, json, glob, hashlib, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/moving', '/tmp/claude-0/v2/detect/py']
import pic, fx
SR = 48000; DRY = '/tmp/claude-0/v2/train/dry/'
KIND = json.load(open('/tmp/claude-0/v2/train/dry_meta.json')); CLIPS = sorted(KIND)
OTHERS = ['flg_calf', 'flg_sox', 'cho_calf', 'cho_sox', 'haas_sox', 'haas_calf_stereotools', 'wid_caps', 'wid_calf_haasenh', 'dly_calf', 'dly_sox', 'dist_pedalboard', 'dist_sox', 'bit_calf',
          'bit_pedalboard', 'trm_sox', 'trm_calf', 'pan_tap', 'pan_calf', 'sat_calf', 'sat_tap_tube', 'exc_calf', 'exc_sox', 'rng_calf', 'rng_sox', 'fil_calf', 'fil_sox', 'stu_sox', 'gat_sox', 'gat_calf']
OUTSIDE = {'swp_pb_ladder': 1, 'swp_pb_bandpass': 1, 'pha_calf': 2, 'pha_swh': 2}; PHASER = float(os.environ.get('PHASER', '0.45'))
OUT_P = np.array([(1 - PHASER) / 2, (1 - PHASER) / 2, PHASER * 0.8, PHASER * 0.2]) if PHASER != 0.45 else None      # with another share: the outside tools likewise, the phaser that is rarely heard less often
def clip(name): return sf.read(DRY + name + '.flac', dtype='float64', always_2d=True)[0]

def one(job):
    out, i, seed = job; r = np.random.default_rng([seed, i]); import zoo_fx as Z
    name = CLIPS[int(r.integers(len(CLIPS)))]; x = clip(name); n = len(x); u = r.random(); info = dict(i=i, clip=name); cls = 0; G = None
    ch = int(r.integers(2)); mid = r.random() < 0.2; pick = (lambda v: v.mean(1)) if mid else (lambda v: v[:, ch]); p = pic.probe(n, seed=i); ps = np.stack([p, p], 1)
    try:
        if u < 0.48:                                             # this study's own moving filter
            xm = pick(x); run, d = (fx.sweep if r.random() < 1.0 - PHASER else fx.phase)(r, xm); y = run(xm); G = pic.curve(p, run(p)); cls = 1 if d['fx'] == 'sweep' else 2; info.update(d); dry = xm
        elif u < 0.64:                                           # an outside tool's moving filter
            nm = str(r.choice(list(OUTSIDE), p=OUT_P)); prm = Z.ZOO[nm][2](r); w = Z.apply(nm, x, prm); wp = Z.apply(nm, ps, prm)
            if mid and nm in ('swp_pb_ladder', 'swp_pb_bandpass'): y = w.mean(1); G = pic.curve(p, wp.mean(1)); dry = x.mean(1)        # the same filter on both sides
            else: y = w[:, ch]; G = pic.curve(p, wp[:, ch]); dry = x[:, ch]
            cls = OUTSIDE[nm]; info.update(fx=nm, params=prm)
        elif u < 0.76: dry = y = pick(x); info.update(fx='plain')
        elif u < 0.80:                                           # a filter that stands still
            xm = pick(x); run, d = (fx.sweep if r.random() < 0.6 else fx.phase)(r, xm * 0.0 + 1.0); f0 = float(np.exp(r.uniform(np.log(d['lo']), np.log(d['hi']))))
            if d['fx'] == 'sweep': y = (1.0 - d['mix']) * xm + d['mix'] * fx.svf(np.ascontiguousarray(xm), np.full(n, f0), d['q'], d['mode'], d['stages'], d['q2'])
            else: y = fx.phaser(np.ascontiguousarray(xm), np.full(n, f0), d['spread'] ** (np.arange(d['stages']) - (d['stages'] - 1) / 2.0), d['fbk'], d['dry'], d['wet'])
            dry = y; info.update(fx='still', of=d['fx'])
        elif u < 0.84:                                           # a level movement: every band moves alike
            xm = pick(x); t = np.arange(n) / SR; rate = float(np.exp(r.uniform(np.log(0.3), np.log(12.0)))); depth = float(r.uniform(0.3, 1.0)); shape = str(r.choice(['sine', 'tri', 'sq']))
            l = np.sign(fx.lfo('sine', rate * t + r.uniform(0, 1))) if shape == 'sq' else fx.lfo(shape, rate * t + r.uniform(0, 1))
            if shape == 'sq': k = int(SR * 0.003); l = np.convolve(np.pad(l, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), mode='valid')
            y = xm * (1.0 - depth * (0.5 + 0.5 * l)); dry = y; info.update(fx='level', rate=rate, depth=depth, shape=shape)
        else:                                                    # another effect family's outside tool
            nm = str(r.choice(OTHERS)); prm = Z.ZOO[nm][2](r); w = Z.apply(nm, x, prm, Z.TAIL.get(Z.ZOO[nm][0], 0.0))[:n]; y = pick(w); dry = y; info.update(fx=nm)
    except Exception as e:
        return None
    if not np.isfinite(y).all() or np.abs(y).max() < 1e-4: return None
    T = n // pic.HOP + 1; c = np.zeros(T, np.uint8)
    if cls:
        if r.random() < 0.25:                                    # the effect on a stretch only
            a = float(r.uniform(0.0, 3.5)); b = float(r.uniform(a + 1.5, 6.5)); t = np.arange(n) / SR; m = np.clip(np.minimum((t - a) / 0.02, (b - t) / 0.02), 0.0, 1.0); y = dry * (1.0 - m) + y * m
            mf = np.interp(np.arange(T) * pic.HOP / SR, t, m); G = G * mf[:, None]; c[mf > 0.5] = cls; info.update(span=[a, min(b, 6.0)])
        else: c[:] = cls
        LEd = pic.picture(dry); LEw = pic.picture(y); top = max(LEd.max(), LEw.max()); live = np.maximum(LEd, LEw) > top - 50.0; heard = float(np.abs(LEw - LEd)[live].mean()) if live.any() else 0.0
        info.update(heard=heard)
        if heard < 1.0: c[:] = 0; G = None; info.update(unheard=True)       # the filter does next to nothing to this sound: nothing to find
    g = 10.0 ** (float(r.uniform(-12.0, 0.0)) / 20.0) * 0.9 / max(np.abs(y).max(), 0.05); LE = pic.picture(y * g); LEd = pic.picture(dry * g)
    return dict(LE=LE.astype(np.float16), G=np.zeros(LE.shape, np.int8) if G is None else np.round(np.clip(G, -40.0, 20.0) * 2.0).astype(np.int8), live=(LEd > LEd.max() - 60.0), cls=c, info=json.dumps(info, default=float))

if __name__ == '__main__':
    out, total = sys.argv[1], int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0; jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 4; os.makedirs(out, exist_ok=True)
    from multiprocessing import Pool
    SH = 250; done = 0
    with Pool(jobs) as pool:
        for s in range(0, total, SH):
            f = '%s/s%d_%05d.npz' % (out, seed, s)
            if os.path.exists(f): continue
            rows = [v for v in pool.map(one, [(out, i, seed) for i in range(s, min(s + SH, total))], chunksize=4) if v is not None]
            np.savez_compressed(f, LE=np.stack([v['LE'] for v in rows]), G=np.stack([v['G'] for v in rows]), live=np.packbits(np.stack([v['live'] for v in rows]), axis=-1), cls=np.stack([v['cls'] for v in rows]), info=np.array([v['info'] for v in rows]))
            done += len(rows); print('%s: %d examples (%d so far)' % (os.path.basename(f), len(rows), done), flush=True)
