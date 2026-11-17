# Lab: for the stored lines that pass everything but the level test, the level change measured only above SR / move (the part
# of the spectrum in which the sweep carries every partial across at least one whole tooth of the comb). Writes levelb.json.
import sys, os, json, pickle, zlib, numpy as np, soundfile as sf
sys.argv = sys.argv[:1]; sys.path.insert(0, '/tmp/claude-0/v2/sweep'); os.chdir('/tmp/claude-0/v2/sweep')
import rule, sweep
def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def hp(x, fc, times=2):
    a = np.exp(-2 * np.pi * fc / 48000.0); y = x
    for _ in range(times):
        import scipy.signal as ss
        y = ss.lfilter([a, -a], [1.0, -a], y)
    return y
def flange(x, rng):
    n = len(x); p = dict(lo=float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))), depth=float(rng.uniform(1.0, 6.0)), rate=float(np.exp(rng.uniform(np.log(0.1), np.log(1.2)))), kind=int(rng.integers(0, 2)),
                         g=float(rng.choice([-1, 1]) * rng.uniform(0.4, 0.95)), fb=float(rng.uniform(-0.6, 0.6)), phase=float(rng.uniform(0, 1)), stereo=float(rng.choice([0.0, 0.25, 0.5])))
    y = np.zeros_like(x); t = np.arange(n) / 48000.0
    for c in range(2):
        d = (p['lo'] + p['depth'] * sweep.lfo_shape(p['rate'] * t + p['phase'] + c * p['stereo'], p['kind'])) * 48.0
        y[:, c] = sweep._run(np.ascontiguousarray(x[:, c]), d, p['g'], p['fb'], False)
    return y
base = rule.make(qlo=-99, qhi=99, reach=64, jumps=0.45); out = {}
for part, rows in (('clean', rule.clean), ('synth', rule.synth)):
    L = pickle.load(open('lines_%s.pkl' % part, 'rb'))
    for r in rows:
        todo = [(c, k, f) for c, cl in enumerate(r['cands']) for k, f in enumerate(cl) if base(f)]
        if not todo: continue
        x = rd('/tmp/claude-0/v2/corpus/dry/%s.wav' % r['id'])
        if part == 'synth': x = flange(x, np.random.default_rng(zlib.crc32(r['id'].encode())))
        same = np.array_equal(x[:, 0], x[:, 1]); e = (x ** 2).sum(0); chans = [c for c in range(2) if not (c == 1 and same) and e[c] >= 1e-4 * e.max()]
        for c, k, f in todo:
            x1 = np.ascontiguousarray(x[:, chans[c]]); l = L[r['id']][c][k]; d = l['d'].astype(float)
            en = (sweep._frames(x1, 1024) ** 2).sum(1); act = en > np.percentile(en, 95) * 1e-5
            o = sweep._run(x1, sweep.per_sample(d, len(x1)), f['g'], f['fb'], True)
            chk = sweep.level_change(x1, o, act); v = {}
            for mult in (0.5, 1.0, 2.0):
                fc = mult * 48000.0 / max(f['move'], 1.0); v['b%g' % mult] = sweep.level_change(hp(x1, fc), hp(o, fc), act)
            v['plain_again'] = chk; out['%s/%s/%d/%d' % (part, r['id'], c, k)] = v
json.dump(out, open('levelb.json', 'w'))
print(len(out), 'lines')
# compare
def q(v, f): return v / f['expect']
for part, rows in (('clean', rule.clean), ('synth', rule.synth)):
    for r in rows:
        for c, cl in enumerate(r['cands']):
            for k, f in enumerate(cl):
                key = '%s/%s/%d/%d' % (part, r['id'], c, k)
                if key not in out or f['fast'] > 0.25: continue
                v = out[key]
                print('%-5s %-24s %s g %+.2f fb %+.2f move %3.0f jumps %.2f | q plain %+.2f (again %+.2f) tilt %+.2f | above SR/move x0.5 %+.2f x1 %+.2f x2 %+.2f %s' % (part, r['id'], f['from'][0], f['g'], f['fb'], f['move'], f['fast'], f['level'] / f['expect'], q(v['plain_again'], f), f['level_t'] / f['expect'],
                      q(v['b0.5'], f), q(v['b1'], f), q(v['b2'], f), ('| derr %5.1f alone %6.1f' % (f['d_err'], f['out_alone'])) if 'd_err' in f else ''))
