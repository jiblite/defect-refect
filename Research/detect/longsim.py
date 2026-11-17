# Longer sounds, as a model only: a long sound taken as k blocks of 6 s, each verified on its own (here: k held-out test
# sounds of one kind standing in for the blocks), and two ways to combine the blocks: an effect is shown when ANY block
# shows it (what the max over windows does today), or when at least HALF of the blocks show it.     python3 longsim.py VARIANT
import sys
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import numpy as np, verifier as V, vreport as R
rows = V.test_rows(); z = np.load(V.D + 'data/test_%s.npz' % sys.argv[1]); B = z['S'] > z['T']; A = R.before(rows); Y = np.array([[f in r['labels'] for f in V.FAMS] for r in rows]); rng = np.random.default_rng(0)
dry = [i for i, r in enumerate(rows) if r['variant'] == 'dry']; kinds = sorted({rows[i]['kind'] for i in dry})
print('clean: k dry test clips of one kind as the blocks of one long sound          before: clean, false/sound | verifier: clean, false/sound')
for k in (1, 2, 4, 8):
    for rule, name in ((lambda M: M.any(0), 'any block'), (lambda M: M.mean(0) >= 0.5, 'half of the blocks')):
        if k == 1 and name != 'any block': continue
        a = []; b = []
        for _ in range(3000):
            kd = kinds[rng.integers(len(kinds))]; pool = [i for i in dry if rows[i]['kind'] == kd]; pick = rng.choice(pool, k); a.append(rule(A[pick]).sum()); b.append(rule(B[pick]).sum())
        a = np.array(a); b = np.array(b); print('  %2d blocks (%2d s), shown when in %-19s %3.0f %%  %.2f | %3.0f %%  %.3f' % (k, 6 * k, name + ':', 100 * np.mean(a == 0), a.mean(), 100 * np.mean(b == 0), b.mean()))
print('held-out effects: k sounds with the same outside effect as the blocks: share found, mean over families        before | verifier')
pos = [i for i, r in enumerate(rows) if r['set'] == 'test' and len(r['labels']) == 1]
for k in (1, 2, 4, 8):
    for rule, name in ((lambda v: v.any(), 'any block'), (lambda v: v.mean() >= 0.5, 'half of the blocks')):
        if k == 1 and name != 'any block': continue
        ra = []; rb = []
        for j, f in enumerate(V.FAMS):
            mine = [i for i in pos if rows[i]['labels'][0] == f]; impls = sorted({rows[i]['impl'][0] for i in mine}); a = []; b = []
            for _ in range(300):
                im = impls[rng.integers(len(impls))]; pool = [i for i in mine if rows[i]['impl'][0] == im]; pick = rng.choice(pool, k); a.append(rule(A[pick, j])); b.append(rule(B[pick, j]))
            ra.append(np.mean(a)); rb.append(np.mean(b))
        print('  %2d blocks, shown when in %-19s %3.0f %% | %3.0f %%' % (k, name + ':', 100 * np.mean(ra), 100 * np.mean(rb)))
