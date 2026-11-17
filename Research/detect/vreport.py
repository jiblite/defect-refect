# Tables for the verification step: the app before it ("before": the detectors' own scores as shown up to 0.4.1, from the
# Python port of the detection, same sounds) against the verifier,
# on the training sounds out of fold (dev) and on the held-out test sounds.     python3 vreport.py VARIANT [dev|test|both]
import sys, json, collections
import paths
import numpy as np, verifier as V
D = V.D; FAMS = V.FAMS
def before(rows): return np.array([[r['final'][V.FIN.index(f)] > V.cut_for(f) for f in FAMS] for r in rows])
def clean_table(rows, shown_a, shown_b, groups, title):
    print('\n' + title); print('  %-34s %4s | before: clean  false/sound | verifier: clean  false/sound' % ('', 'n'))
    for name, sel in groups:
        i = np.array([sel(r) for r in rows])
        if not i.any(): continue
        a = shown_a[i].sum(1); b = shown_b[i].sum(1); print('  %-34s %4d | %10.0f %% %10.2f | %14.0f %% %10.2f' % (name, i.sum(), 100 * np.mean(a == 0), a.mean(), 100 * np.mean(b == 0), b.mean()))
def effect_table(rows, shown_a, shown_b, Y, clean, pos_sel, title, by_impl=True):
    print('\n' + title); print('  %-12s | present: n  before  verifier | absent on clean sounds: before  verifier | absent on other effects\' sounds: before  verifier' % 'effect')
    ra = []; rb = []
    for k, f in enumerate(FAMS):
        p = Y[:, k] & pos_sel; o = (~Y[:, k]) & pos_sel & (Y.sum(1) > 0)
        if not p.any(): continue
        ra.append(shown_a[p, k].mean()); rb.append(shown_b[p, k].mean())
        print('  %-12s | %10d %5.0f %% %8.0f %% | %27.1f %% %8.1f %% | %38.1f %% %8.1f %%' % (f, p.sum(), 100 * ra[-1], 100 * rb[-1], 100 * shown_a[clean, k].mean(), 100 * shown_b[clean, k].mean(),
              100 * shown_a[o, k].mean() if o.any() else 0, 100 * shown_b[o, k].mean() if o.any() else 0))
    print('  mean over families: found before %.0f %%, verifier %.0f %%' % (100 * np.mean(ra), 100 * np.mean(rb)))
    if by_impl:
        print('  per outside effect (found before -> verifier / sounds):')
        for k, f in enumerate(FAMS):
            E = collections.OrderedDict()
            for i, r in enumerate(rows):
                if pos_sel[i] and Y[i, k]:
                    for a, b in zip(r['labels'], r['impl']):
                        if a == f: E.setdefault(b, []).append((shown_a[i, k], shown_b[i, k]))
            print('    %-12s' % f + '  '.join('%s %d->%d/%d' % (n, sum(v[0] for v in e), sum(v[1] for v in e), len(e)) for n, e in sorted(E.items())))
def dev(variant, budget=0.07, floor=0.5):
    rows = V.train_rows(); S = np.load(D + 'data/oof_%s.npz' % variant)['S']; assert len(S) == len(rows)
    Y = np.array([[f in r['labels'] for f in FAMS] for r in rows]); clean = Y.sum(1) == 0; T, ks = V.thresholds(S, Y, clean, budget, floor); A = before(rows); B = S > T
    print('=== DEV, out of fold (%s): %d training sounds from %d sources, %d clean; cuts from a budget of %.2f false effects per clean sound' % (variant, len(rows), len({r['group'] for r in rows}), clean.sum(), budget))
    print('  cuts:', ' '.join('%s %.2f' % (f[:6], t) for f, t in zip(FAMS, T)))
    G = [('all clean training sounds', lambda r: True)] + [(v, (lambda r, v=v: r['variant'] == v)) for v in ('raw', 'mono', 'narrow', 'bal', 'seg', 'speed', 'rev', 'mix', 'pad', 'cat', 'codec', 'reverb_ir', 'tone_ir', 'case_reverb', 'case_tone')]
    cr = [r for r, c in zip(rows, clean) if c]; clean_table(cr, A[clean], B[clean], G, 'clean training sounds by how they were made')
    kinds = sorted({r['kind'] for r in cr}); clean_table(cr, A[clean], B[clean], [(k, (lambda r, k=k: r['kind'] == k)) for k in kinds], 'clean training sounds by kind of source')
    own = np.array([len(r['labels']) > 0 and all(i == 'own' for i in r['impl'][:len(r['labels'])]) for r in rows]); single = Y.sum(1) == 1
    effect_table(rows, A, B, Y, clean, single & own, 'the app\'s own effects, alone on a sound', by_impl=False)
    effect_table(rows, A, B, Y, clean, single & ~own, 'outside effects (dev split), alone on a sound, scored by models that never saw that effect')
    effect_table(rows, A, B, Y, clean, Y.sum(1) == 2, 'two effects on a sound', by_impl=False)
    return T
def test(variant):
    rows = V.test_rows(); z = np.load(D + 'data/test_%s.npz' % variant); S = z['S']; T = z['T']; assert len(S) == len(rows)
    Y = np.array([[f in r['labels'] for f in FAMS] for r in rows]); A = before(rows); B = S > T; held = np.array([r['set'] == 'test' for r in rows]); clean = (Y.sum(1) == 0) & held
    print('\n=== HELD-OUT TEST (%s): test-split clips, test-split outside effects; nothing here was used for fitting or for the cuts' % variant)
    G = [('dry test clips', lambda r: r['variant'] == 'dry'), ('  of them real recordings', lambda r: r['variant'] == 'dry' and r['kind'] == 'real-recording'), ('reverb only (test reverbs)', lambda r: r['variant'] == 'case_reverb'), ('tone only (test tone effects)', lambda r: r['variant'] == 'case_tone'),
         ('THE THREE TOGETHER', lambda r: r['variant'] in ('dry', 'case_reverb', 'case_tone')), ('dry + 3 s silence', lambda r: r['variant'] == 'drypad'), ('dry as 16-bit', lambda r: r['variant'] == 'dry16'), ('dry through a lossy codec', lambda r: r['variant'] == 'drymp3'),
         ('two test clips mixed', lambda r: r['variant'] == 'drymix')]
    cr = [r for r, c in zip(rows, clean) if c]; clean_table(cr, A[clean], B[clean], G, 'clean test sounds')
    main = np.array([r['variant'] in ('dry', 'case_reverb', 'case_tone') for r in rows]) & clean
    kinds = sorted({r['kind'] for r in cr}); clean_table([r for r, c in zip(rows, main) if c], A[main], B[main], [(k, (lambda r, k=k: r['kind'] == k)) for k in kinds], 'clean test sounds (dry, reverb only, tone only) by kind of source')
    stored = np.array([bool(r.get('stored')) for r in rows])
    effect_table(rows, A, B, Y, main, held & (Y.sum(1) > 0), 'held-out positives: stored scoreboard cases (8 per family) and fresh settings on test clips', by_impl=True)
    effect_table(rows, A, B, Y, main, held & stored & (Y.sum(1) > 0), 'of them the stored scoreboard cases only (test half, 8 sounds per family)', by_impl=False)
    pd = np.array([r['set'] == 'posdev' for r in rows])
    if pd.any(): effect_table(rows, A, B, Y, main, pd, 'for comparison, NOT held out: the scoreboard\'s dev half (these outside effects were in the training, on other settings)', by_impl=False)
    p = held & (Y.sum(1) > 0); print('\n  other effects shown on a held-out positive sound: before %.2f per sound, verifier %.2f' % ((A[p] & ~Y[p]).sum(1).mean(), (B[p] & ~Y[p]).sum(1).mean()))
if __name__ == '__main__':
    what = sys.argv[2] if len(sys.argv) > 2 else 'both'
    if what in ("dev", "both"): dev(sys.argv[1], float(sys.argv[3]) if len(sys.argv) > 3 else 0.07, float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
    if what in ('test', 'both'): test(sys.argv[1])

# ---- one line per method: the app before, thresholds alone (re-chosen cuts on the detectors' scores), the verifier variants
def base_scores(rows, how):
    """The detectors' numbers as scores: 'final' = the app's final scores (max over windows + rules); 'mean' / 'min' = the deciders' scores over the windows."""
    if how == 'final': return np.array([[r['final'][V.FIN.index(f)] for f in FAMS] for r in rows], np.float32)
    S = np.zeros((len(rows), len(FAMS)), np.float32)
    for i, r in enumerate(rows):
        if r['nwin'] == 0: continue
        for k, f in enumerate(FAMS):
            p = r['p27'][:, V.F27.index(f)] if f in V.NEW else r['p12'][:, V.F12.index(f)]; S[i, k] = p.mean() if how == 'mean' else p.min()
    return S
def summary(variants, budget=0.07, floor=0.5):
    tr = V.train_rows(); te = V.test_rows(); Ytr = np.array([[f in r['labels'] for f in FAMS] for r in tr]); Yte = np.array([[f in r['labels'] for f in FAMS] for r in te]); ctr = Ytr.sum(1) == 0
    held = np.array([r['set'] == 'test' for r in te]); grp = lambda v: np.array([r['variant'] == v for r in te]) & held; main = (grp('dry') | grp('case_reverb') | grp('case_tone')); pos = held & (Yte.sum(1) > 0)
    own = np.array([len(r['labels']) > 0 and all(i == 'own' for i in r['impl'][:len(r['labels'])]) for r in tr]); single = Ytr.sum(1) == 1
    def rec(B, Y, sel): return 100 * np.mean([B[sel & Y[:, k], k].mean() for k in range(len(FAMS)) if (sel & Y[:, k]).any()])
    def line(name, Btr, Bte):
        f = lambda B, s: '%3.0f %% %.2f' % (100 * np.mean(B[s].sum(1) == 0), B[s].sum(1).mean())
        print('  %-44s | %s  %3.0f %%  %3.0f %% | %s | %s | %s | %s |  %3.0f %%   %.2f' % (name, f(Btr, ctr), rec(Btr, Ytr, single & own), rec(Btr, Ytr, single & ~own), f(Bte, grp('dry')), f(Bte, grp('case_reverb')), f(Bte, grp('case_tone')), f(Bte, main),
              rec(Bte, Yte, pos), (Bte[pos] & ~Yte[pos]).sum(1).mean()))
    print('\n=== SUMMARY (cuts for every method but the first: chosen on the training sounds for %.2f false effects per clean sound)' % budget)
    print('  %-44s | %-31s | %-10s | %-10s | %-10s | %-10s | %s' % ('method', 'TRAINING sounds: clean false/snd', 'TEST dry', 'reverb only', 'tone only', 'all three', 'held-out effects found, other effects per positive sound'))
    print('  %-44s | %-31s | %-10s | %-10s | %-10s | %-10s |' % ('', '   found: own  outside(unseen)', 'clean f/s', 'clean f/s', 'clean f/s', 'clean f/s'))
    line('before: 0.4.1 at its default', before(tr), before(te))
    for how, nm in (('final', 'thresholds alone: new cuts on the detectors\' scores'), ('mean', 'windows averaged (not max) + new cuts'), ('min', 'lowest window + new cuts')):
        Str = base_scores(tr, how); T, _ = V.thresholds(Str, Ytr, ctr, budget, floor=0.0); line(nm, Str > T, base_scores(te, how) > T)
    for v in variants:
        try: S = np.load(D + 'data/oof_%s.npz' % v)['S']; z = np.load(D + 'data/test_%s.npz' % v)
        except FileNotFoundError: continue
        T, _ = V.thresholds(S, Ytr, ctr, budget, floor); line('verifier: ' + v, S > T, z['S'] > T)
if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[2] == 'summary': summary(sys.argv[1].split(','), float(sys.argv[3]) if len(sys.argv) > 3 else 0.07, float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
