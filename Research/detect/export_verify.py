# Writes the verification models as the app reads them (Source/Core/Verify.h Verifier::load), and checks that trees read
# back from the file give the same scores as the Python models.
#   python3 export_verify.py VARIANT [--install]      ->  work/model/verify_VARIANT.bin  (--install: also Resources/verify.bin)
# Layout (little-endian): 'DRVF', version u32 (2), the inputs' names (count u32, then length u32 + bytes each), the effects
# (count u32), then per effect: its name, z65 f32 (the raw tree output that is shown as 65 %: the fixed cut), slope f32 (of
# the shown score, calib.py), and the trees as Models.h Boosted reads them (baseline f32 array, number of trees u32, per tree:
# number of nodes u32, nodes of feature i32, threshold f32, left i32, right i32, leaf u8, missing-goes-left u8, value f32).
# The slope is one number for all effects: the median of the logistic fits of "the effect is there" against the raw score on
# the out-of-fold scores. Fitted per effect it goes from 0.17 (chorus) to 0.82 (haas), and the flat ones make the slider
# useless on the training sounds (at 50 % one clean sound in seven would show a chorus); with the common slope 86 % of the
# clean training sounds stay clean at 50 % and 97 % at 80 %.
import sys, struct, pickle, shutil
import paths
import numpy as np, calib
D = paths.WORK
def slope_for(variant):
    import verifier as V
    rows = V.train_rows(); S = np.load(D + 'data/oof_%s.npz' % variant)['S']; assert len(S) == len(rows); Y = np.array([[f in r['labels'] for f in V.FAMS] for r in rows])
    per = [calib.platt(calib.logit(S[:, k]), Y[:, k])[0] for k in range(len(V.FAMS))]; return float(np.median(per)), per
def write(variant, path):
    paths.make_work(); M = pickle.load(open(D + 'model/verifier_%s.pkl' % variant, 'rb')); slope, per = slope_for(variant); out = open(path, 'wb')
    u32 = lambda v: out.write(struct.pack('<I', v)); i32 = lambda v: out.write(struct.pack('<i', v)); f32 = lambda v: out.write(struct.pack('<f', float(v))); s = lambda t: (u32(len(t.encode())), out.write(t.encode()))
    def f32a(a): a = np.asarray(a, np.float32).ravel(); u32(len(a)); out.write(a.tobytes())
    out.write(b'DRVF'); u32(2); u32(len(M['names']))
    for n in M['names']: s(n)
    u32(len(M['fams']))
    for f, m, c in zip(M['fams'], M['models'], M['cuts']):
        s(f); f32(calib.logit(c)); f32(slope); f32a(np.asarray(m._baseline_prediction).ravel()); u32(len(m._predictors))
        for it in m._predictors:
            for p in it:
                n = p.nodes; u32(len(n))
                for r in n:
                    i32(int(r['feature_idx'])); f32(r['num_threshold']); i32(int(r['left'])); i32(int(r['right']))
                    out.write(struct.pack('<B', int(r['is_leaf']))); out.write(struct.pack('<B', int(r['missing_go_to_left']))); f32(r['value'])
    out.close(); return M, slope, per
NODE = np.dtype([('feat', '<i4'), ('thr', '<f4'), ('left', '<i4'), ('right', '<i4'), ('leaf', 'u1'), ('miss', 'u1'), ('value', '<f4')])
class Family:
    """One effect's trees as read from the file. raw(x): the trees' output for one row of inputs (float32); shown(z): the score the app shows."""
    def __init__(s, name, z65, slope, base, trees): s.name = name; s.z65 = z65; s.slope = slope; s.base = base; s.trees = trees
    def raw(s, x):
        o = s.base
        for t in s.trees:
            n = 0
            while not t['leaf'][n]:
                v = x[t['feat'][n]]; n = (t['left'][n] if t['miss'][n] else t['right'][n]) if np.isnan(v) else (t['left'][n] if v <= t['thr'][n] else t['right'][n])
            o += float(t['value'][n])
        return o
    def shown(s, z): return float(calib.shown(z, s.z65, s.slope))
def read(path):
    b = open(path, 'rb').read(); p = [4]; assert b[:4] == b'DRVF'
    def u32(): v = struct.unpack_from('<I', b, p[0])[0]; p[0] += 4; return v
    def f32(): v = struct.unpack_from('<f', b, p[0])[0]; p[0] += 4; return v
    def st(): n = u32(); v = b[p[0]:p[0] + n].decode(); p[0] += n; return v
    assert u32() == 2; names = [st() for _ in range(u32())]; fams = []
    for _ in range(u32()):
        name = st(); z65 = f32(); slope = f32(); nb = u32(); base = np.frombuffer(b, '<f4', nb, p[0]); p[0] += 4 * nb; assert nb == 1; trees = []
        for _ in range(u32()): n = u32(); trees.append(np.frombuffer(b, NODE, n, p[0])); p[0] += NODE.itemsize * n
        fams.append(Family(name, z65, slope, float(base[0]), trees))
    return names, fams
if __name__ == '__main__':
    import os, hashlib, verifier as V
    variant = sys.argv[1]; path = D + 'model/verify_%s.bin' % variant; M, slope, per = write(variant, path); names, fams = read(path)
    print('slope of the shown score: %.3f (median of the per-effect fits: %s)' % (slope, ' '.join('%s %.2f' % (f[:6], a) for f, a in zip(M['fams'], per))))
    rows = V.test_rows()[::7]; X, _, _ = V.matrices(rows, variant); worst = 0.0; flips = 0
    for fam, m, c in zip(fams, M['models'], M['cuts']):
        a = np.array([fam.raw(np.asarray(x, np.float32)) for x in X]); b = m.predict_proba(X)[:, 1]; pa = 1 / (1 + np.exp(-a)); worst = max(worst, float(np.abs(pa - b).max()))
        flips += int(((np.array([fam.shown(z) for z in a]) > 0.65) != (b > c)).sum())
    print('%s: %.1f MB, %d inputs, %d effects, sha256 %s' % (path, os.path.getsize(path) / 1e6, len(names), len(fams), hashlib.sha256(open(path, 'rb').read()).hexdigest()))
    print('trees read back from the file against the Python models on %d test sounds: largest score difference %.2g; shown at 65 %% (file) against above the cut (Python): %d of %d decisions differ' % (len(rows), worst, flips, len(rows) * len(fams)))
    if '--install' in sys.argv: shutil.copyfile(path, paths.RES + 'verify.bin'); print('copied to', paths.RES + 'verify.bin')
