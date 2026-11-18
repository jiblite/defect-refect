# What should the Amount fader's far left be? The model's answer g (dB per band and frame, 0 .. -30) reshaped in several
# ways, each scored against the true direct part over the whole file, over the sound itself (the first 6 s) and over the
# ring-out after it (the last 3 s), on the scoreboard's 120 reverb cases.
#   b      g' = -30 * sigmoid(logit(-g/30) + b): the doubtful cells move most (b = 1.5 is the far left of 0.4.1 to 0.4.5)
#   k      g' = k * g: plain scaling of the answer in dB (k = 0.5 is the fader half way to the answer)
#   sharp  p' = p + c * p^2 * (1 - p), p = -g/30: cells the model is sure about go deeper, doubtful ones stay
#   deep   g' = g * (1 + (m - 1) * step((-g - 6) / 12)): the same idea with a threshold (6 to 18 dB)
#   MODEL=unet_x+unet_xp2_e5 NP=4 python3 depth2_test.py [step]    -> depth2_test_<MODEL>.json
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, score, render, predict
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_x+unet_xp2_e5'); SR = 48000
def m_b(b): return lambda g: -30.0 / (1.0 + np.exp(-(np.log(np.clip(-g / 30.0, 1e-6, 1 - 1e-6) / (1 - np.clip(-g / 30.0, 1e-6, 1 - 1e-6))) + b)))
def m_k(k): return lambda g: np.maximum(k * g, -30.0)
def m_sharp(c): return lambda g: -30.0 * np.clip(-g / 30.0 + c * (-g / 30.0) ** 2 * (1 + g / 30.0), 0.0, 1.0)
def m_deep(m):
    def f(g): u = np.clip((-g - 6.0) / 12.0, 0.0, 1.0); return np.maximum(g * (1.0 + (m - 1.0) * u * u * (3 - 2 * u)), -30.0)
    return f
def m_con(lo, hi, a, w):
    """Contrast: cells the answer leaves shallow are scaled by lo, cells it takes deep by hi; the change-over is from a to a + w dB."""
    def f(g): u = np.clip((-g - a) / w, 0.0, 1.0); return np.maximum(g * (lo + (hi - lo) * u * u * (3 - 2 * u)), -30.0)
    return f
if os.environ.get('SET') == '2':
    MAPS = [('answer', lambda g: g), ('b -1', m_b(-1.0)), ('b -0.5', m_b(-0.5)), ('k 0.6', m_k(0.6))] + [('c%.1f/%.1f@%d+%d' % (lo, hi, a, w), m_con(lo, hi, a, w)) for lo in (0.5, 0.7) for hi in (1.0, 1.5, 2.0) for a, w in ((3, 9), (6, 12))]
else: MAPS = [('b -1', m_b(-1.0)), ('answer', lambda g: g), ('b 0.5', m_b(0.5)), ('b 1.5', m_b(1.5)), ('k 0.5', m_k(0.5)), ('k 0.75', m_k(0.75)), ('k 1.25', m_k(1.25)), ('k 1.5', m_k(1.5)),
        ('sharp 1', m_sharp(1.0)), ('sharp 2', m_sharp(2.0)), ('sharp 3', m_sharp(3.0)), ('deep 1.5', m_deep(1.5)), ('deep 2', m_deep(2.0)), ('deep 3', m_deep(3.0))]
def cells(ref, b, floor_db=-60.0, cap_db=30.0):
    """As score.distance, cell by cell: per resolution (difference per frame, the frame's middle in samples)."""
    out = []
    for (n_fft, hop, n_mels), PA, PB in zip(score.RES, score.pictures(ref), score.pictures(b[:len(ref)])):
        top = PA.max() + 1e-20; LA = 10 * np.log10(np.maximum(PA, top * 10 ** (floor_db / 10))); LB0 = 10 * np.log10(np.maximum(PB, 1e-30)); live = LA > LA.min() + 6
        g = np.median((LA - LB0)[live]) if live.any() else 0.0; dd = np.minimum(np.abs(LA - np.maximum(LB0 + g, LA.min())), cap_db); out.append((dd.mean((0, 1)), np.arange(dd.shape[-1]) * hop + n_fft // 2))
    return out
def seg(C, a, b): return float(np.mean([d[(t >= a) & (t < b)].mean() for d, t in C]))
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; n = len(wet); L = 6 * SR        # every scoreboard clip is 6 s long, the file is the clip and 3 s after it
    ft = cm.features(wet, direct); g = predict.gains(ft, MODEL)
    Cw = cells(direct, wet); base = [seg(Cw, 0, n), seg(Cw, 0, L), seg(Cw, L, n)]
    r = dict(name=c['name'], split=c['meta']['split'], drr=float(10 * np.log10((direct ** 2).sum() / max((rev ** 2).sum(), 1e-20))), base=base, body_len=L / SR, sc={})
    for nm, f in MAPS:
        C = cells(direct, render.render(wet, f(g), 0.0)); r['sc'][nm] = [100.0 * (1.0 - seg(C, a, b) / max(w, 1e-9)) for (a, b), w in zip(((0, n), (0, L), (L, n)), base)]
    return r
def show(rows):
    print('model %s, %d cases (the sound itself: %.1f s of %.1f s)' % (MODEL, len(rows), rows[0]['body_len'], 9.0)); drr = np.array([r['drr'] for r in rows])
    for part, pn in enumerate(('WHOLE FILE', 'THE SOUND ITSELF (without the ring-out)', 'THE RING-OUT')):
        print(pn); print('  %-9s | %-28s | %-28s | %-28s' % ('', 'all (n=%d)' % len(rows), 'heavy, direct < 5 dB over (n=%d)' % (drr < 5).sum(), 'light (n=%d)' % (drr >= 5).sum()))
        for nm, _ in MAPS:
            S = np.array([r['sc'][nm][part] for r in rows]); cols = []
            for sel in (np.ones(len(rows), bool), drr < 5, drr >= 5): cols.append('med %5.1f mean %6.1f worse %3.0f%%' % (np.median(S[sel]), S[sel].mean(), 100 * (S[sel] < 0).mean()))
            print('  %-9s | %s | %s | %s' % (nm, cols[0], cols[1], cols[2]))
    for part, pn in enumerate(('whole', 'sound itself', 'ring-out')):
        B = np.array([[r['sc'][nm][part] for nm, _ in MAPS] for r in rows]); best = B.argmax(1); print('best map per case, %s: typical %.1f, average %.1f; how often each is best: %s' % (pn, np.median(B.max(1)), B.max(1).mean(), ', '.join('%s %d' % (MAPS[j][0], (best == j).sum()) for j in range(len(MAPS)) if (best == j).any())))
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--show': show(json.load(open('/tmp/claude-0/v2/train/depth2_test_%s%s.json' % (MODEL, os.environ.get('SET', ''))))); sys.exit(0)
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '1'))) as p: rows = p.map(one, ds, chunksize=1)
    json.dump(rows, open('/tmp/claude-0/v2/train/depth2_test_%s%s.json' % (MODEL, os.environ.get('SET', '')), 'w')); show(rows); print('%.0f s' % (time.time() - t0))
