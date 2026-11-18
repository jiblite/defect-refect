# Is a "voice" the sound's own pitch? For each voice of a sound: where on its round its peaks lie (share of frames with a
# peak, by eighth of the round), and whether its peaks have partners in the same frame at simple ratios of its delay
# (a pitched sound shows its period at every multiple; a chorus voice stands alone, or has a weaker one at twice the delay if fed back).
#   python3 t24.py clean:<id> | <case>:A|B | x:<x00_A> ...
import sys, json, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
import chorus as C, engine as E
from lab import load, SETS
RATIOS = [(2, 1), (3, 1), (1, 2), (1, 3), (3, 2), (2, 3), (4, 3), (3, 4), (4, 1), (1, 4), (5, 4), (4, 5), (5, 3), (3, 5), (5, 2), (2, 5)]
def sound(a):
    k, name = a.split(':', 1) if ':' in a else ('case', a)
    if k == 'clean':
        x = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % name, dtype='float64', always_2d=True)[0]; return (x if x.shape[1] > 1 else np.repeat(x, 2, 1))[:, :2]
    if k == 'x': return sf.read('/dev/shm/chx/%s_wet.wav' % name, dtype='float64', always_2d=True)[0]
    return load(SETS + 'chorus/' + k, name)['wet']
def family(P, v, m):
    """For the voice's peaks (mask m): per ratio, the share of them with a partner peak in the same frame at ratio * delay (stretch scaled alike), and the partners' mean height against the voice's."""
    out = {}; byt = {}
    for i in range(len(P)): byt.setdefault(int(P[i, 0]), []).append(i)
    idx = np.where(m)[0]
    for a, b in RATIOS:
        q = a / b; n = 0; hs = []; can = 0
        for i in idx:
            d = P[i, 1] * q; r = P[i, 2] * q
            if not (C.TMIN + 2 <= d <= C.TMAX - 2): continue
            can += 1; best = None
            for j in byt[int(P[i, 0])]:
                if j != i and abs(P[j, 1] - d) < 2.0 * max(q, 1.0) * C.TOL and abs(P[j, 2] - r) < 2.0 * max(q, 1.0) * C.RTOL: best = P[j, 3] / P[i, 3] if best is None else max(best, P[j, 3] / P[i, 3])
            if best is not None: n += 1; hs.append(best)
        if can >= 0.3 * len(idx) and can > 0: out['%d/%d' % (a, b)] = (n / can, float(np.mean(hs)) if hs else 0.0, can / max(len(idx), 1))
    return out
for a in sys.argv[1:]:
    y = sound(a); print(a if 'real' not in a else 'clean:(one of the recordings)', flush=True)
    for ch in range(2):
        if ch == 1 and np.array_equal(y[:, 0], y[:, 1]): print('  side 1 = side 0'); continue
        A, act = C.picture(y[:, ch]); T = len(A); P = C.peaks(A, act); del A; info = []; V = C.voices(P, T, need=E.NEED, info=info); fw = len(np.unique(P[:, 0]))
        for v in V:
            m = C.near(P, v); ph = (v['rate'] * P[m, 0] / C.FPS + v['phase']) % 1.0; fr = np.unique(P[m, 0]); phf = (v['rate'] * fr / C.FPS + v['phase']) % 1.0
            allf = np.unique(P[:, 0]); pha = (v['rate'] * allf / C.FPS + v['phase']) % 1.0; share = [float(((phf * 8).astype(int) == k).sum()) / max(((pha * 8).astype(int) == k).sum(), 1) for k in range(8)]
            print('  side %d %s %.4f Hz %.2f+-%.2f ms on %.2f height %.1f | on by eighth of the round: %s  (least %.2f)' % (ch, 'tri' if v['kind'] else 'sin', v['rate'], v['mid'] / C.FS * 1e3, abs(v['half']) / C.FS * 1e3, v['on'], v['height'], ' '.join('%.2f' % s for s in share), min(share)))
            fam = family(P, v, m); print('      partners (share of the peaks, height against the voice): ' + '  '.join('%s %.2f/%.2f' % (k, s, h) for k, (s, h, c) in fam.items() if s >= 0.15))
