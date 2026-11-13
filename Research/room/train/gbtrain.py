# The quick model: boosted trees on hand-made clues, trained on the training cases, checked on the scoreboard's reverb cases.
import sys, glob, json, time, os, pickle
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, gbfeat, net
from sklearn.ensemble import HistGradientBoostingRegressor
from multiprocessing import Pool
PER = int(os.environ.get('PER', '1500'))
def rows(f):
    z = np.load(f); LE, LS = z['LE'].astype(np.float64), z['LS'].astype(np.float64); X = gbfeat.features(LE, z['U64'].astype(np.float64), z['U256'].astype(np.float64)); g, live = gbfeat.target(LE, LS)
    idx = np.argwhere(live); rng = np.random.default_rng(abs(hash(f)) % 2 ** 32); sel = idx[rng.choice(len(idx), min(PER, len(idx)), replace=False)]
    p, base = net.proxy(np.zeros_like(LE), LE, LS); w = 1.0 / max(base, 1.0)          # a cell counts more in a sound with little reverb: there the same error is a larger share of the whole
    return X[sel[:, 0], sel[:, 1]], g[sel[:, 0], sel[:, 1]].astype(np.float32), np.full(len(sel), w, np.float32)
def sb_eval(model, keep=None):
    out = []
    for f in sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz')):
        z = np.load(f); info = json.loads(str(z['info'])); LE, LS = z['LE'].astype(np.float64), z['LS'].astype(np.float64); X = gbfeat.features(LE, z['U64'].astype(np.float64), z['U256'].astype(np.float64))
        if keep is not None: X = X[..., keep]
        g = np.clip(model.predict(X.reshape(-1, X.shape[-1])).reshape(LE.shape), -30.0, 0.0); p, base = net.proxy(g, LE, LS); gt, live = gbfeat.target(LE, LS); pt, _ = net.proxy(gt, LE, LS)
        out.append((os.path.basename(f)[:-4], info['split'], info['kind'], info['drr'], p, pt, base))
    return out
def report(out, title):
    a = np.array([[r[3], r[4], r[5]] for r in out]); sp = np.array([r[1] for r in out]); print(title)
    for lab, sel in [('scoreboard, reverbs also used in training (other settings, other sounds)', sp == 'dev'), ('scoreboard, reverbs and sounds never seen in training', sp == 'test')]:
        for lab2, s2 in [('all', np.ones(len(a), bool)), ('light (DRR >= 8)', a[:, 0] >= 8), ('middle', (a[:, 0] < 8) & (a[:, 0] >= 0)), ('wet (DRR < 0)', a[:, 0] < 0)]:
            s = sel & s2
            if s.sum(): print('  %-70s %-18s n=%3d  model: median %5.1f mean %5.1f, below zero in %2.0f%% | with the true answer: median %5.1f' % (lab if lab2 == 'all' else '', lab2, s.sum(), np.median(a[s, 1]), np.mean(a[s, 1]), 100 * np.mean(a[s, 1] < 0), np.median(a[s, 2])))
if __name__ == '__main__':
    t0 = time.time(); files = sorted(glob.glob('/tmp/claude-0/v2/train/cases/*.npz')); print(len(files), 'training cases')
    with Pool(int(os.environ.get('NP', '4'))) as p: parts = p.map(rows, files, chunksize=8)
    X = np.concatenate([a for a, b, c in parts]); y = np.concatenate([b for a, b, c in parts]); W = np.concatenate([c for a, b, c in parts]) if os.environ.get('WEIGHT', '1') == '1' else None; print(X.shape, 'rows, %.0f s' % (time.time() - t0), 'target mean %.2f dB' % y.mean())
    m = HistGradientBoostingRegressor(loss='absolute_error', max_iter=int(os.environ.get('IT', '300')), learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100, early_stopping=False, random_state=0)
    m.fit(X, y, sample_weight=W); print('trained, %.0f s' % (time.time() - t0)); pickle.dump(m, open(os.environ.get('SAVE', '/tmp/claude-0/v2/train/gb.pkl'), 'wb'))
    out = sb_eval(m); report(out, 'quick model'); json.dump(out, open(os.environ.get('SAVE', '/tmp/claude-0/v2/train/gb.pkl') + '.sb.json', 'w'))
    from sklearn.inspection import permutation_importance
    r = permutation_importance(m, X[::200], y[::200], n_repeats=2, random_state=0, scoring='neg_mean_absolute_error'); o = np.argsort(-r.importances_mean)
    print('most useful clues:', ', '.join('%s %.2f' % (gbfeat.NAMES[i], r.importances_mean[i]) for i in o[:16])); print('%.0f s' % (time.time() - t0))
