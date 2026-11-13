#!/usr/bin/env python3
"""Scores the TONE engine prototype (tone.py) on the scoreboard: all 64 tone cases, the check on clean sounds, Tone Match.

  nice python3 tone_eval.py              runs everything with 2 processes (several minutes), saves tone_eval.json, prints the tables
  python3 tone_eval.py --show            prints the tables again from tone_eval.json
  options: --jobs N (default 2), --clean N (clean training clips for the harm check, default 200, 0 = none), --only dev|test

What is measured. d(a, b) is the scoreboard's distance (bench/score.py). A = the sound that is analysed, B = the second sound.
  Removed %   100 * (1 - d(A dry, A wet with the curve taken out) / d(A dry, A wet))          the fader at 0
  Refect %    100 * (1 - d(B wet, B dry with A's curve put on)   / d(B wet, B dry))           the second sound's fader at 1
              100 = perfect, 0 = nothing gained, below zero = worse than doing nothing.
  default     the curve comes from tone.contour(A wet) as shipped (reflections only): nothing but the wet sound is known
  with model  tone.contour(A wet, model=True): the model's guess added (restraint 2); 'model, no restraint': the same, not shrunk
  ceiling     the same filter (16384 taps, minimum phase, same caps) with the TRUE curve of the effect, as fine as the engine's own
              curve (one value per analysis bin, 5.86 Hz). 'per chan.' = left and right each get their own true curve: what a perfect
              blind estimate would reach. 'one curve' = the two channels' true curves averaged into one, the engine's default layout.
  harm        d(dry, flatten(dry)): the engine's fader at 0 on a sound that has no effect on it. Zero = untouched.
  Tone Match  tone.match(target = A dry, sound = A wet, Detail D) put on A wet, scored like Removed: both versions of the sound are known.
"""
import sys, os, json, time, glob, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, '/tmp/claude-0/v2/proto', '/tmp/claude-0/v2/bench']
import numpy as np
SETS = '/tmp/claude-0/v2/bench/sets/tone/'; CLEAN = '/tmp/claude-0/v2/train/dry/tr_*.flac'; OUT = os.path.join(HERE, 'tone_eval.json')
# the settings that are compared: name -> arguments of tone.contour. The first one is the engine's default.
SETTINGS = {'default': {}, 'with model': dict(model=True), 'model, no restraint': dict(model=True, restraint=0.0)}
MATCH = (0.6, 1.0)                                             # Detail values for the Tone Match rows

def true_curve(c, n=1 << 17):
    """The effect's true magnitude curve in dB from its click responses, per channel, as means over the engine's bins: (2, bins)."""
    import tone
    H = np.stack([np.fft.rfft(c['h'][o][:n, o], n) for o in range(2)]); db = 20 * np.log10(np.maximum(np.abs(H), 1e-7)); k = n // tone.N; out = np.empty((2, len(tone.FB)))
    for o in range(2):
        cs = np.concatenate([[0.0], np.cumsum(db[o])]); i = np.arange(len(tone.FB)) * k; lo = np.clip(i - k // 2, 0, len(db[o])); hi = np.clip(i + k // 2 + 1, 0, len(db[o])); out[o] = (cs[hi] - cs[lo]) / (hi - lo)
    return out

def flat(c): return float(np.abs(c).max()) < 1e-9

def case(d):
    import truth, score, tone
    a = truth.load(d, 'A'); b = truth.load(d, 'B'); FB = tone.FB
    sa = score.Scorer(a['dry']); sb = score.Scorer(b['wet']); sbd = score.Scorer(b['dry']); dwa = sa(a['wet']); dwb = sb(b['dry'])
    rm = lambda c: 100 * (1 - sa(tone.apply_curve(a['wet'], FB, c, 0.0)) / dwa); rf = lambda c: 100 * (1 - sb(tone.put_curve(b['dry'], FB, c, 1.0)) / dwb)
    row = dict(kind='case', name=a['name'], effect=a['meta']['effect'], split=a['meta']['split'], dwa=dwa, dwb=dwb, rm={}, rf={}, harmA={}, harmB={}, found={}, size={})
    for name, kw in SETTINGS.items():
        info = []; g, c = tone.contour(a['wet'], info=info, **kw); row['rm'][name] = 0.0 if flat(c) else rm(c); row['rf'][name] = 0.0 if flat(c) else rf(c)
        row['found'][name] = [(float(dl), float(gn)) for dl, gn in info[0]['echoes']]; row['size'][name] = float(np.abs(info[0]['model_db']).max())
        for key, x, s in (('harmA', a['dry'], sa), ('harmB', b['dry'], sbd)):
            g, c = tone.contour(x, **kw); row[key][name] = 0.0 if flat(c) else float(s(tone.apply_curve(x, FB, c, 0.0)))
    t = true_curve(a); t1 = t.mean(0)
    row['rm']['ceiling, one curve'] = rm(t1); row['rf']['ceiling, one curve'] = rf(t1); row['rm']['ceiling, per chan.'] = rm(t); row['rf']['ceiling, per chan.'] = rf(t)
    for det in MATCH:
        g, c = tone.match(a['dry'], a['wet'], det); row['rm']['Tone Match, D %g' % det] = 100 * (1 - sa(tone.put_curve(a['wet'], g, c, 1.0)) / dwa)
    return row

def clean(p):
    import soundfile as sf, score, tone
    x = sf.read(p, always_2d=True)[0].astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    x = x[:, :2]; s = score.Scorer(x); row = dict(kind='clean', harm={}, found={})
    for name, kw in SETTINGS.items():
        info = []; g, c = tone.contour(x, info=info, **kw); row['harm'][name] = 0.0 if flat(c) else float(s(tone.apply_curve(x, tone.FB, c, 0.0))); row['found'][name] = len(info[0]['echoes'])
    return row

def work(job): return case(job[1]) if job[0] == 'case' else clean(job[1])

def stats(v): v = np.asarray(v, float); return '%6.1f %6.1f %4.0f%%' % (np.median(v), v.mean(), 100 * (v < 0).mean())

def show(rows):
    cases = [r for r in rows if r['kind'] == 'case']; cl = [r for r in rows if r['kind'] == 'clean']; names = list(cases[0]['rm']); sets = list(cases[0]['harmA'])
    groups = [('%s, all %d' % (sp, sum(r['split'] == sp for r in cases)), [r for r in cases if r['split'] == sp]) for sp in ('dev', 'test')]
    for e in sorted(set(r['effect'] for r in cases)): rs = [r for r in cases if r['effect'] == e]; groups.append(('%s (%s)' % (e.replace('tone_', ''), rs[0]['split']), rs))
    groups = [g for g in groups if g[1]]
    for key, title in (('rm', 'REMOVED % (fader at 0 on the analysed sound)'), ('rf', 'REFECT % (the curve found on A put onto the second sound B)')):
        cols = [n for n in names if n in cases[0][key]]; print('\n' + title + ': median, mean, share of cases below zero'); print('%-24s' % '' + ''.join(' | %-19s' % n[:19] for n in cols))
        for lab, rs in groups: print('%-24s' % lab + ''.join(' | ' + stats([r[key][n] for r in rs]) for n in cols))
    print('\nHARM ON CLEAN SOUNDS: d(dry, flatten(dry)) at the fader\'s 0, next to what an effect does, d(dry, wet)')
    for sp in ('dev', 'test'):
        rs = [r for r in cases if r['split'] == sp]
        if not rs: continue
        dw = np.array([r['dwa'] for r in rs] + [r['dwb'] for r in rs]); print('  %s: the %d dry sounds of the cases. For comparison d(dry, wet): median %.2f, mean %.2f, smallest %.2f' % (sp, 2 * len(rs), np.median(dw), dw.mean(), dw.min()))
        for n in sets:
            h = np.array([r['harmA'][n] for r in rs] + [r['harmB'][n] for r in rs]); print('     %-20s untouched (exactly 0): %3d of %d | mean %.3f, median %.3f, 90th percentile %.3f, largest %.3f | over 0.1: %d' % (n, (h == 0).sum(), len(h), h.mean(), np.median(h), np.percentile(h, 90), h.max(), (h > 0.1).sum()))
    if cl:
        print('  %d clean training clips (no effect on them)' % len(cl))
        for n in sets:
            h = np.array([r['harm'][n] for r in cl]); f = np.array([r['found'][n] for r in cl]); print('     %-20s untouched (exactly 0): %3d of %d | mean %.3f, median %.3f, 90th percentile %.3f, largest %.3f | over 0.1: %d | a reflection reported on %d' % (n, (h == 0).sum(), len(h), h.mean(), np.median(h), np.percentile(h, 90), h.max(), (h > 0.1).sum(), (f > 0).sum()))
    print('\nALL CASES. dw = d(A dry, A wet); removed and refect: default | with model | ceiling one curve | ceiling per channel; harm of the default on A dry, B dry; reflections found on A wet as delay in samples : gain')
    for r in cases:
        print('%-22s %-4s dw %5.2f | removed %6.1f | %6.1f | %5.1f | %5.1f || refect %6.1f | %6.1f | %5.1f | %5.1f || harm %.3f %.3f || %s' % (r['name'], r['split'], r['dwa'], r['rm']['default'], r['rm']['with model'], r['rm']['ceiling, one curve'], r['rm']['ceiling, per chan.'],
              r['rf']['default'], r['rf']['with model'], r['rf']['ceiling, one curve'], r['rf']['ceiling, per chan.'], r['harmA']['default'], r['harmB']['default'], ' '.join('%.2f:%+.2f' % tuple(e) for e in r['found']['default']) or '-'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--show', action='store_true'); ap.add_argument('--jobs', type=int, default=2); ap.add_argument('--clean', type=int, default=200); ap.add_argument('--only', default='')
    ap.add_argument('--out', default=OUT); ap.add_argument('--every', type=int, default=1, help='take every n-th case only (quick trial)'); o = ap.parse_args()
    if not o.show:
        from multiprocessing import Pool
        t0 = time.time(); ds = sorted(glob.glob(SETS + '*/'))
        if o.only: ds = [d for d in ds if json.load(open(d + 'meta.json'))['split'] == o.only]
        ds = ds[::o.every]
        files = sorted(glob.glob(CLEAN)); files = [files[i] for i in np.random.default_rng(123).permutation(len(files))[:o.clean]]
        jobs = [('case', d) for d in ds] + [('clean', p) for p in files]
        if o.jobs > 1:
            with Pool(o.jobs) as pool: rows = pool.map(work, jobs, chunksize=1)
        else: rows = [work(j) for j in jobs]
        json.dump(rows, open(o.out, 'w')); print('%d cases, %d clean clips, %.0f s' % (len(ds), len(files), time.time() - t0))
    show(json.load(open(o.out)))
