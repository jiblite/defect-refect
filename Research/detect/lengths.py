# Sounds of other lengths than the 6 to 9 s of the held-out test: what the verification gives on them, on held-out
# material only.
#   python3 lengths.py short     held-out sounds cut to 1 s, 2 s and 4 s (taken from 2 s in)
#   python3 lengths.py long      30 s sounds joined from 5 held-out clips: dry, through a held-out reverb or tone case, and
#                                with a held-out outside effect on all of it (one setting for the whole sound)
#   python3 lengths.py report    the tables (results/lengths_short.jsonl, results/lengths_long.jsonl)
# All in Python (appdet.py for the detectors, evidence.py, the trees of verify.bin, the blocks of vref.py). As in the app
# the whole sound goes through the detectors once and every block is judged from the windows that lie inside it. Every
# block's scores are kept, so the first k blocks of a long sound give the result for a sound of k blocks.
import sys, json, os, time
import paths
import numpy as np
SR = 48000; what = sys.argv[1]; RESULTS = paths.WORK + 'results/'
def half(v): v = sorted(v, reverse=True); return v[(len(v) + 1) // 2 - 1] if v else 0.0       # the rule of Verify.h: the highest score that at least half of the blocks reach
if what == 'report':
    import appdet, verifier as V, export_verify as XV
    FAMS = appdet.SHOWN; rd = lambda n: [json.loads(l) for l in open(RESULTS + n)] if os.path.exists(RESULTS + n) else []
    def table(rows, shown, title):
        """rows: (kind, labels, id); shown(row) -> set of effects. Clean share, false per sound, found (mean over the effects), other effects per sound with an effect."""
        cl = [r for r in rows if not r['labels']]; po = [r for r in rows if r['labels']]; n = [len(shown(r)) for r in cl]
        per = [np.mean([f in shown(r) for r in po if r['labels'] == [f]]) for f in FAMS if any(r['labels'] == [f] for r in po)]
        return '%-34s clean sounds (%3d): %3.0f %% clean, %.2f false per sound | with an effect (%3d): %3.0f %% found, %.2f other effects per sound' % (
            title, len(cl), 100 * np.mean([k == 0 for k in n]) if cl else 0, np.mean(n) if cl else 0, len(po), 100 * np.mean(per) if per else 0, np.mean([len(shown(r) - set(r['labels'])) for r in po]) if po else 0)
    after = lambda r: { f for f in FAMS if r['final'][f] > 0.65 }; before = lambda r: set(appdet.shown(r['before']))
    S = rd('lengths_short.jsonl')
    if S:
        print('SHORT SOUNDS (held-out sounds cut from 2 s in; up to 3 s the detectors see one window, filled up with silence)')
        names, fams = XV.read(paths.RES + 'verify.bin'); FN = V.feat_names(); col = [FN.index(n) for n in names]; have = { r['id']: r for r in V.test_rows() }; ids = { r['id'].split('/', 1)[1].rsplit('/', 1)[0] for r in S }
        full = [dict(labels=have[i]['labels'], final={ f.name: f.shown(f.raw(V.vec(have[i])[col])) for f in fams }, before=dict(zip(V.FIN, map(float, have[i]['final'])))) for i in sorted(ids) if i in have]
        print('  ' + table(full, before, 'whole (6 to 9 s), before:')); print('  ' + table(full, after, 'whole (6 to 9 s), verifier:'))
        for sec in (4, 2, 1):
            rows = [r for r in S if r['sec'] == sec]; print('  ' + table(rows, before, '%d s, before:' % sec)); print('  ' + table(rows, after, '%d s, verifier:' % sec))
        for sec in (4, 2, 1):
            rows = [r for r in S if r['sec'] == sec and r['labels']]
            print('  %d s, found per effect, before -> verifier: %s' % (sec, '  '.join('%s %.0f -> %.0f' % (f, 100 * np.mean([f in before(r) for r in rows if r['labels'] == [f]]), 100 * np.mean([f in after(r) for r in rows if r['labels'] == [f]])) for f in FAMS)))
    L = rd('lengths_long.jsonl')
    if L:
        print('LONG SOUNDS (30 s, joined from 5 held-out clips; the first k blocks of each stand for a sound of k blocks)')
        print('  ' + table(L, before, 'the whole 30 s, before:'))
        def first(k, rule):
            def shown(r):
                it = { f: iter(r['blocks'][f]) for f in FAMS }; per = { f: [next(it[f]) for j in r['judged'][:k] if j] for f in FAMS }       # 'blocks' holds the judged blocks only, in order
                return { f for f in FAMS if rule(per[f]) > 0.65 }
            return shown
        median = lambda v: float(np.median(v)) if v else 0.0
        for k in (1, 2, 3, 4, 5):
            for name, rule in (('half of the blocks (the app)', half), ('any block', lambda v: max(v, default=0.0)), ('every block', lambda v: min(v, default=0.0)), ('the median score', median)):
                if (k == 1 and name != 'half of the blocks (the app)') or (k % 2 == 1 and name == 'the median score'): continue       # an odd number of blocks: the median is the app's rule
                print('  ' + table(L, first(k, rule), '%d block%s, %s:' % (k, 's' if k > 1 else '', name if k > 1 else 'as one sound')))
        print('  ' + table(L, lambda r: { f for f in FAMS if r['one12'][f] > 0.65 }, 'the first 12 s as ONE block:'))
        def pairs(rule):         # every two blocks of a sound as a sound of two blocks (ten per sound)
            cl = []; po = []
            for r in L:
                if not all(r['judged']): continue
                for a in range(5):
                    for b in range(a + 1, 5):
                        sh = { f for f in FAMS if rule([r['blocks'][f][a], r['blocks'][f][b]]) > 0.65 }
                        if r['labels']: po.append(r['labels'][0] in sh)
                        else: cl.append(len(sh))
            return '%3.0f %% clean, %.2f false per sound | %3.0f %% found' % (100 * np.mean([c == 0 for c in cl]), np.mean(cl), 100 * np.mean(po))
        for name, rule in (('the higher one (the app)', max), ('the lower one', min), ('their mean', lambda v: 0.5 * (v[0] + v[1]))): print('  two blocks, every pair of blocks of a sound, %-26s %s' % (name + ':', pairs(rule)))
    sys.exit(0)

import gen_feats as G, vref, export_verify as XV, zoo_fx as Z, appdet, evidence as EV
paths.make_work(); os.makedirs(RESULTS, exist_ok=True); model = XV.read(paths.RES + 'verify.bin'); outp = RESULTS + 'lengths_%s.jsonl' % what
done = { json.loads(l)['id'] for l in open(outp) } if os.path.exists(outp) else set(); out = open(outp, 'a'); t0 = time.time()
def judge(y, single=False):
    A = appdet.analyze(y); bl = [(0, len(y))] if single else vref.blocks(len(y)); per = []
    for b0, ln in bl:
        rows = vref.block_windows(A['starts'], b0, ln)
        if not rows: per.append(None); continue
        yb = y[b0:b0 + ln]; echo = A['echo']
        if len(bl) > 1: sc, ms = appdet._echo.echo_detect(yb); echo = (float(sc), float(ms or 0))
        per.append(vref.inputs(appdet.OLD, vref.F27, A['logit12'][rows], A['hand'][rows], A['p12'][rows], A['logit27'][rows], A['p27'][rows], echo, EV.evidence(yb)))
    fin, blocks = vref.score_blocks(model, per); return fin, blocks, A['final'], [x is not None for x in per]
def put(id_, kind, labels, y, **kw):
    if id_ in done: return
    y = np.ascontiguousarray(np.asarray(y, np.float64).astype(np.float32).astype(np.float64)); fin, blocks, before, judged = judge(y)
    if len(judged) > 1: kw['one12'] = judge(y[:12 * SR], True)[0]            # the first 12 s taken as one block, against the rule's two
    out.write(json.dumps(dict(id=id_, kind=kind, labels=labels, n=len(y), final=fin, blocks=blocks, before=before, judged=judged, **kw)) + '\n'); out.flush()
if what == 'short':
    items = [('dry', 'dry/' + c, G.dry(c), []) for c in G.TEST]
    for fam in ('reverb', 'tone'): items += [(fam + ' only', '%s/%s/%s' % (fam, os.path.basename(d[:-1]), w), G.case_sound(d, w)[0], []) for d in G.cases(fam, 'test') for w in 'AB']
    for fam in G.FAMS: items += [('effect', '%s/%s/%s' % (fam, os.path.basename(d[:-1]), w), G.case_sound(d, w)[0], [fam]) for d in G.cases(fam, 'test') for w in 'AB']
    captured = []; G.feats = lambda y: (captured.append(np.asarray(y, np.float64)), dict(n=len(y)))[1]           # the fresh held-out effect sounds of the test run (a quarter of them), made again
    for fam in G.FAMS:
        captured.clear(); rows = G.test_ext_rows(fam, 0); assert len(rows) == len(captured)
        items += [('effect', r['id'], y, [fam]) for r, y in zip(rows, list(captured))]
    print('%d sounds, %.0f s' % (len(items), time.time() - t0), flush=True)
    for i, (kind, name, y, labels) in enumerate(items):
        for sec in (1, 2, 4):
            seg = y[2 * SR:(2 + sec) * SR]
            if len(seg) < sec * SR or np.sqrt((seg ** 2).mean()) < 1e-3: continue
            put('%s/%s/%ds' % (kind, name, sec), kind, labels, G.fade(seg), sec=sec)
        if i % 40 == 0: print('short', i, len(items), '%.0f s' % (time.time() - t0), flush=True)
else:
    r = np.random.default_rng(31); T = G.TEST
    def join(k=5): return np.vstack([G.fade(G.dry(str(c))) for c in r.choice(T, k, replace=False)])
    for i in range(24): put('dry/%02d' % i, 'dry', [], join())
    for fam in ('reverb', 'tone'):
        cs = G.cases(fam, 'test')
        for i in range(12): d = cs[(i * 5) % len(cs)]; put('%s only/%02d' % (fam, i), fam + ' only', [], G.fade(G.through_ir(join(), d, 3.0 if fam == 'reverb' else 0.5)[:30 * SR]))
    print('long clean done, %.0f s' % (time.time() - t0), flush=True)
    for fam in G.FAMS:
        names = G.ZTEST[fam]
        for i in range(3):
            name = names[i % len(names)]; rr = np.random.default_rng(G.h32('long/%s/%d' % (name, i))); clipf = Z.ZOO[name][4]; ok = None
            for _ in range(8):
                x = join()
                if clipf is not None and not clipf(x): continue
                try: p = Z.ZOO[name][2](rr); w = Z.apply(name, x, p, 0.0)[:len(x)]
                except Exception as e: continue
                if np.isfinite(w).all() and np.abs(w).max() > 1e-3 and G.change_db(x, w) > -30: ok = w; break
            if ok is None: print('no audible', fam, name, i, flush=True); continue
            put('effect/%s/%s_%d' % (fam, name, i), 'effect', [fam], ok, impl=name)
        print('long', fam, '%.0f s' % (time.time() - t0), flush=True)
print('finished', flush=True)
