# The stutter engine (stutter.py) on the scoreboard's stutter cases: the events found against the events made, and the
# map put onto sound B against B through the real effect.   python3 su1.py
import sys, os, json, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx'); sys.path.insert(0, '/home/claude/defect-refect/Research/detect')
from st_common import *
import stutter as E
def plan(n, p):
    r = np.random.default_rng(p['seed']); L = int(p['slice_ms'] * 1e-3 * SR); pos = 0; out = []
    while pos + L <= n:
        if r.random() < p['prob']: out.append((pos, L, p['repeats'])); pos += p['repeats'] * L
        else: pos += L
    return out
for d in cases('stutter'):
    m = json.load(open(d + 'meta.json')); A = truth.load(d, 'A'); B = truth.load(d, 'B'); t0 = time.time(); f = E.find(A['wet']); dt = time.time() - t0; n = len(A['wet']); L0 = int(m['params']['slice_ms'] * 1e-3 * SR)
    # where the cut-out sound starts in what the effect was run on: the shift that makes the most found events fall on made ones
    P0 = plan(n + 4 * SR, m['params']); best = (0, int(m['pre'] * SR))
    sh = best[1]; P = [(a - sh, b, a - sh + b * c) for a, b, c in P0 if a - sh + b * c > 0 and a - sh < n]; got = set(f['events']); want = set((max(a, 0) if a >= 0 else a, b, min(c, n)) for a, b, c in P)
    print('%-14s slice %d samples x %d, the cut starts %d samples in: %d events made, %d found, %d of them right; fades %s; share %.2f; %.2f s' % (os.path.basename(d[:-1]), L0, m['params']['repeats'], sh, len(P), len(got), len(got & want), f.get('fades'), f.get('share', 0), dt))
    for e in sorted(got - want)[:4]: print('      found but not made:', e)
    for e in sorted(want - got)[:4]: print('      made but not found:', e, '(the slice is silent in A)' if e[0] >= 0 and np.abs(A['wet'][e[0]:e[0] + e[1]]).max() < 1e-4 else '')
    if f['found']:
        o = E.onto(B['dry'], f); nb = min(len(o), len(B['wet'])); print('      onto B: %.1f of 100, waveform %.1f dB beside B through the real effect' % (score.removed(B['wet'][:nb], B['dry'][:nb], o[:nb])[0], db(o[:nb] - B['wet'][:nb], B['wet'][:nb])))
        o2 = E.play(B['dry'], [(max(a, 0), b, min(c, nb)) for a, b, c in P if a >= 0], f['fades']); print('      the made events put onto B the same way: %.1f of 100, waveform %.1f dB' % (score.removed(B['wet'][:nb], B['dry'][:nb], o2[:nb])[0], db(o2[:nb] - B['wet'][:nb], B['wet'][:nb])))
