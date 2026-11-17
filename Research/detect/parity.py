# The C++ (drverify, built from Tests/verify.cpp: the app's two detectors and Source/Core/Verify.h) against the Python of
# this folder, on sounds that are made here, written as 32-bit float WAV, measured by both and deleted.
#   python3 parity.py bulk path/to/drverify    about 300 held-out sounds of the test run (clean, stored effect cases, fresh
#                                              effects), made again with the same seeds. Two comparisons per sound:
#                                              'py'   all in Python: the feature rows the held-out results come from
#                                                     (gen_feats.py test), scored with the trees of verify.bin; where a sound
#                                                     does not come out as in that run (an outside effect that starts at a
#                                                     random phase) the Python side measures the new sound;
#                                              'dump' everything behind the detectors: the trees and the evidence in Python
#                                                     on what the C++ detectors gave per window (vref.from_dump)
#   python3 parity.py long path/to/drverify    five sounds of 30 s and more, joined from held-out clips: every block through
#                                              the Python detectors and evidence.py (vref.verify_sound)
#   python3 parity.py report                   the summary of the bulk run (results/parity_bulk.jsonl)
# Tools/verify_parity.py does the same for any WAV files.
import sys, json, subprocess, os, time
import paths
import numpy as np
what = sys.argv[1]; RESULTS = paths.WORK + 'results/'; TMP = paths.WORK + 'tmp/par/'; CUT = 0.65
if what == 'report':
    R = [json.loads(l) for l in open(RESULTS + 'parity_bulk.jsonl')]; kinds = {}
    for r in R: k = 'clean' if not r['labels'] else 'stored effect cases' if r['variant'].startswith('case_') else 'fresh effects'; kinds[k] = kinds.get(k, 0) + 1
    print('%d sounds (%s); for %d of them the Python detectors ran anew (the sound did not come out as in the feature run)' % (len(R), ', '.join('%d %s' % (v, k) for k, v in kinds.items()), sum(r['src'] == 'fresh' for r in R)))
    for key, title in (('dump', 'everything behind the detectors (trees and evidence in Python on the C++ detectors\' numbers)'), ('py', 'all in Python (the detectors\' Python port too)')):
        flips = [(r['id'], k) for r in R for k in r[key] if (r[key][k] > CUT) != (r['cpp'][k] > CUT)]; dif = sorted(max(abs(r[key][k] - r['cpp'][k]) for k in r[key]) for r in R)
        print('%s: shown / not shown at 65 %%: %d of %d decisions differ; largest difference of a score %.2g (sounds with a difference above 1e-6: %d, above 0.01: %d)' % (title, len(flips), sum(len(r[key]) for r in R), dif[-1], sum(v > 1e-6 for v in dif), sum(v > 0.01 for v in dif)))
        for i, k in flips: print('   differs:', i, k)
    print('evidence, C++ against evidence.py on the same sound: largest relative difference %.2g' % max(r['evdiff'] for r in R))
    print('the C++ shows an effect on %d of the %d clean sounds (before: on %d)' % (sum(any(v > CUT for v in r['cpp'].values()) for r in R if not r['labels']), sum(not r['labels'] for r in R),
          sum(any(v > (0.8 if k in ('saturation', 'exciter', 'ringmod', 'filter', 'stutter', 'gatedrhythm') else CUT) for k, v in r['before'].items() if k in r['cpp']) for r in R if not r['labels'])))
    sys.exit(0)
import soundfile as sf
import gen_feats as G, verifier as V, vref, export_verify as XV, zoo_fx as Z
drverify = sys.argv[2]; paths.make_work(); os.makedirs(RESULTS, exist_ok=True); os.makedirs(TMP, exist_ok=True); model = XV.read(paths.RES + 'verify.bin'); names, fams = model
def cpp(y, tag):
    p = TMP + '%s_%d.wav' % (tag, os.getpid()); sf.write(p, y.astype(np.float32), 48000, subtype='FLOAT')
    try: o = subprocess.run([drverify, paths.RES, p], capture_output=True, text=True)
    finally: os.remove(p)
    return json.loads(o.stdout) if o.returncode == 0 else o.stderr[:300]
if what == 'bulk':
    captured = []; G.feats = lambda y: (captured.append(np.asarray(y, np.float64)), dict(n=len(y)))[1]
    FN = V.feat_names(); col = [FN.index(n) for n in names]; have = { r['id']: r for r in V.test_rows() }; outp = RESULTS + 'parity_bulk.jsonl'
    U = dict(G.units('test')); plan = [('t_dry_%02d' % i, 1) for i in (0, 4, 8)] + [('t_cases_%02d' % i, 1) for i in (0, 3, 6)] + [('t_mix', 2)] + [('t_pos_%s_test' % f, 2) for f in G.FAMS] + [('t_ext_%s_%d' % (f, q), 3) for q in (0, 1) for f in G.FAMS]
    done = { json.loads(l)['id'] for l in open(outp) } if os.path.exists(outp) else set(); out = open(outp, 'a'); t0 = time.time(); k = 0
    for unit, stride in plan:
        captured.clear(); rows = U[unit](); assert len(rows) == len(captured), unit
        for r, y in list(zip(rows, captured))[::stride]:
            if r['id'] in done or r['id'] not in have: continue
            d = cpp(y, 'b')
            if isinstance(d, str): print('C++ FAILED', r['id'], d, flush=True); continue
            h = have[r['id']]; assert len(d['blocks']) == 1; y32 = y.astype(np.float32).astype(np.float64); dump, _, evs = vref.from_dump(d, y32, model); cev = d['blocks'][0]['ev']
            evd = max(abs(evs[0][n] - cev[n]) / (1e-9 + max(abs(evs[0][n]), abs(cev[n]))) for n in cev) if evs[0] is not None else 0.0          # the C++ evidence against evidence.py on this very sound
            x = V.vec(h)[col]; py = { f.name: f.shown(f.raw(x)) for f in fams }; ev = dict(zip(V.EVN, h['ev'])); src = 'rows'
            if h['n'] != d['n'] or evs[0] is None or max(abs(float(ev[n]) - evs[0][n]) / (1e-4 + max(abs(float(ev[n])), abs(evs[0][n]))) for n in cev) > 1e-4:
                py, _ = vref.verify_sound(y32, model); src = 'fresh'        # the sound did not come out as in the feature run: the Python detectors on this very sound
            out.write(json.dumps(dict(id=r['id'], variant=r['variant'], labels=r['labels'], n=d['n'], src=src, cpp=d['final'], py=py, dump=dump, before=d['before'], evdiff=evd, ms=d['ms'], nwin=len(d['win']))) + '\n'); out.flush(); k += 1
        print('%s done: %d sounds, %.0f s' % (unit, k, time.time() - t0), flush=True)
    print('finished', flush=True)
else:
    r = np.random.default_rng(2026); T = G.TEST
    def join(names): return np.vstack([G.fade(G.dry(c)) for c in names])
    def fx(name, x, seed):
        rr = np.random.default_rng(seed)
        for _ in range(8):
            p = Z.ZOO[name][2](rr); w = Z.apply(name, x, p, 0.0)
            if np.isfinite(w).all() and np.abs(w).max() > 1e-3 and G.change_db(x, w) > -30: return w[:len(x)]
        raise RuntimeError('no audible ' + name)
    pick = lambda k: [str(c) for c in r.choice(T, k, replace=False)]; S = []
    S.append(('5 dry clips joined, 30 s', join(pick(5)), []))
    x = join(pick(6)); S.append(('6 clips, 36 s, a tremolo (outside, test split) on all of it', fx('trm_tap', x, 1), ['tremolo']))
    x = join(pick(5)); w = x.copy(); w[18 * 48000:] = fx('pan_mda', x[18 * 48000:], 2); S.append(('5 clips, 30 s, auto pan on the last 12 s only (2 of 5 blocks)', w, []))
    x = join(pick(6))[:int(33.5 * 48000)]; S.append(('33.5 s (5 blocks, the last one 9.5 s), haas on all of it', fx('haas_ffmpeg', x, 3), ['haas']))
    a = G.dry(pick(1)[0]); b = fx('rng_mda', join(pick(3)), 4); S.append(('6 s dry, 9 s of silence, 18 s with a ring modulator, 2.5 s of silence (35.5 s)', np.vstack([a, np.zeros((9 * 48000, 2)), b, np.zeros((int(2.5 * 48000), 2))]), ['ringmod']))
    worst = 0.0; flips = 0; nd = 0
    for i, (title, y, truth) in enumerate(S):
        y = np.ascontiguousarray(y.astype(np.float32).astype(np.float64)); d = cpp(y, 'long'); assert not isinstance(d, str), d
        t0 = time.time(); py, per = vref.verify_sound(y, model); tp = time.time() - t0; dump, per2, evs = vref.from_dump(d, y, model)
        bl = vref.blocks(len(y)); assert [(b['start'], b['len']) for b in d['blocks']] == bl
        evd = max((abs(e[k] - b['ev'][k]) / (1e-9 + max(abs(e[k]), abs(b['ev'][k]))) for b, e in zip(d['blocks'], evs) if e is not None for k in e), default=0.0)
        dif = max(abs(py[k] - d['final'][k]) for k in py); dif2 = max(abs(dump[k] - d['final'][k]) for k in py); bdif = max((abs(a - c) for k in per for a, c in zip(per[k], d['scores'][k])), default=0.0)
        fl = [k for k in py if (py[k] > CUT) != (d['final'][k] > CUT)]; worst = max(worst, dif); flips += len(fl); nd += len(py)
        print('%d. %s: %.1f s, %d blocks %s' % (i + 1, title, len(y) / 48000, len(bl), [round(b[1] / 48000, 1) for b in bl]))
        print('   judged blocks %d; C++ against Python: final scores differ by at most %.2g (block scores %.2g; from the C++ windows %.2g; evidence, relative %.2g); decisions that differ: %s' % (len(next(iter(d['scores'].values()))), dif, bdif, dif2, evd, fl or 'none'))
        print('   shown at 65 %%: C++ %s | Python %s | on the sound: %s | before (the rule up to 0.4.1): %s' % (sorted(k for k in py if d['final'][k] > CUT), sorted(k for k in py if py[k] > CUT), truth, sorted(k for k in py if d['before'][k] > (0.8 if k in vref.NEW else CUT))))
        print('   per block, effects above 65 %%: %s' % [sorted(k for k in per if per[k][j] > CUT) for j in range(len(next(iter(per.values()))))])
        print('   CPU time: C++ detectors %.1f s, verification %.2f s (evidence %.2f s, echo checks %.2f s); all in Python %.0f s of wall clock' % (d['ms']['inspect'] / 1e3, d['ms']['verify'] / 1e3, d['ms']['evidence'] / 1e3, d['ms']['echo'] / 1e3, tp), flush=True)
    print('ALL: largest difference of a final score %.2g; decisions that differ %d of %d' % (worst, flips, nd))
