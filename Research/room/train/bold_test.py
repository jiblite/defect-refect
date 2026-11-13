# The Amount fader's far left: how does the real score move when the model's answer is made bolder (depth b in
# g' = -30 * sigmoid(logit(-g/30) + b))? Per case: score of "reverb out", and what happens to tails and to the body in dB.
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, score, render, predict
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_b9'); BS = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
def curve(g, b):
    p = np.clip(-g / 30.0, 1e-6, 1 - 1e-6); return -30.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; ft = cm.features(wet, direct); g = predict.gains(ft, MODEL); sc0 = score.Scorer(direct); dw = sc0(wet)
    E = 10 ** (ft['LE'] / 10); live = ft['LE'] > ft['LE'].max() - 50.0; room = live & (ft['LR'] - ft['LS'] > 6.0); body = live & (ft['LS'] - ft['LR'] > 6.0)
    r = dict(name=c['name'], base=float(dw), drr=float(10 * np.log10((direct ** 2).sum() / max((rev ** 2).sum(), 1e-20))), sc=[], tail=[], body=[])
    for b in BS:
        gb = curve(g, b); G = 10 ** (gb / 10); cut = lambda sel: float(-10 * np.log10(max((E * G)[sel].sum(), 1e-30) / max(E[sel].sum(), 1e-30))) if sel.any() else float('nan')
        r['sc'].append(100.0 * (1.0 - sc0(render.render(wet, gb, 0.0)) / max(dw, 1e-9))); r['tail'].append(cut(room)); r['body'].append(cut(body))
    return r
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 6]
    with Pool(int(os.environ.get('NP', '1'))) as p: rows = p.map(one, ds)
    json.dump(rows, open('/tmp/claude-0/v2/train/bold_test_%s.json' % MODEL, 'w')); SC = np.array([r['sc'] for r in rows]); TL = np.array([r['tail'] for r in rows]); BD = np.array([r['body'] for r in rows]); drr = np.array([r['drr'] for r in rows]); base = np.array([r['base'] for r in rows])
    print('model %s, %d cases' % (MODEL, len(rows))); print('depth b:                    ' + ' '.join('%6.1f' % b for b in BS))
    print('score, typical case:        ' + ' '.join('%6.1f' % v for v in np.median(SC, 0))); print('score, average:             ' + ' '.join('%6.1f' % v for v in np.mean(SC, 0)))
    print('score, pooled:              ' + ' '.join('%6.1f' % (100.0 * (1.0 - (base * (1.0 - SC[:, j] / 100.0)).sum() / base.sum())) for j in range(len(BS))))
    print('made worse (%):             ' + ' '.join('%6.0f' % (100 * v) for v in np.mean(SC < 0, 0)))
    print('room on its own down by dB: ' + ' '.join('%6.1f' % v for v in np.nanmedian(TL, 0))); print('sound on top loses dB:      ' + ' '.join('%6.2f' % v for v in np.nanmedian(BD, 0)))
    for lab, sel in (('wet (DRR < 0)', drr < 0), ('middle', (drr >= 0) & (drr < 8)), ('light (DRR >= 8)', drr >= 8)):
        if sel.sum(): print('%-18s typical:  ' % lab + ' '.join('%6.1f' % v for v in np.median(SC[sel], 0)) + '   n=%d' % sel.sum())
    print('best depth per case: typical score %.1f, average %.1f' % (np.median(SC.max(1)), np.mean(SC.max(1)))); print('%.0f s' % (time.time() - t0))
