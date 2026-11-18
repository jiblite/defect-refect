# Where along a sound the reverb engine helps and where it harms: the score of long_score.py / long_once.py taken
# stretch by stretch (every 6 s of the body, and the ring-out after it), for a looped long sound and for the same clip
# with the same reverb played once. The whole-file score is a mean over all cells, so a ring-out that is 3 s of 9 s
# weighs a third there and 3 s of 39 s a thirteenth: this tells a real effect of length from that.
#   python3 long_parts.py path/to/drrender path/to/Resources tag name [name ...]    -> prints; keeps the renders in tmp/parts
import sys, os, json, re, subprocess
sys.dont_write_bytecode = True; sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/roomsure'); sys.path.insert(0, '/home/claude/defect-refect/Research/room/bench')
import numpy as np, soundfile as sf, scipy.signal as ss
import zoo as R, prep_long as PL, score
D = '/tmp/claude-0/v2/roomsure/'; SR = 48000; CLICK = 480; P = D + 'tmp/parts/'
EXE, RES, TAG = sys.argv[1:4]; T = {r['name']: r for r in json.load(open(D + 'long_truth.json'))}
def cells(ref, b, floor_db=-60.0, cap_db=30.0):
    """As score.distance, but the differences cell by cell: a list of (difference per frame, the frame's middle in samples) per resolution."""
    n = min(len(ref), len(b)); out = []
    for (n_fft, hop, n_mels), PA, PB in zip(score.RES, score.pictures(ref[:n]), score.pictures(b[:n])):
        top = PA.max() + 1e-20; LA = 10 * np.log10(np.maximum(PA, top * 10 ** (floor_db / 10))); LB0 = 10 * np.log10(np.maximum(PB, 1e-30)); live = LA > LA.min() + 6
        g = np.median((LA - LB0)[live]) if live.any() else 0.0; dd = np.minimum(np.abs(LA - np.maximum(LB0 + g, LA.min())), cap_db)
        out.append((dd.mean((0, 1)), np.arange(dd.shape[-1]) * hop + n_fft // 2))
    return out
def seg(C, a, b):
    return float(np.mean([d[(t >= a) & (t < b)].mean() for d, t in C if ((t >= a) & (t < b)).any()]))
def render(tagname, pi):
    p1, p2 = P + tagname + '_half.wav', P + tagname + '_left.wav'
    if not (os.path.exists(p1) and os.path.exists(p2)):
        q = subprocess.run([EXE, RES, pi, '-', p1, 'reverb.amount=0.5', '--', '-', p2, 'reverb.amount=0'], capture_output=True, text=True)
        if q.returncode: raise RuntimeError(q.stdout[-300:] + q.stderr[-300:])
        open(P + tagname + '.txt', 'w').write(q.stdout)
    return p1, p2
def table(lab, wet, direct, p1, p2, edges):
    rd = lambda f: sf.read(f, dtype='float64', always_2d=True)[0][:len(wet)]
    Cw, Ch, Cl = cells(direct, wet), cells(direct, rd(p1)), cells(direct, rd(p2)); n = len(wet)
    print('  %s: whole file half way left %6.1f, far left %7.1f' % (lab, 100 * (1 - seg(Ch, 0, n) / seg(Cw, 0, n)), 100 * (1 - seg(Cl, 0, n) / seg(Cw, 0, n))))
    for a, b, nm in edges:
        w = seg(Cw, a, b); print('     %-22s untouched %5.2f dB off | half way left %6.1f | far left %7.1f' % (nm, w, 100 * (1 - seg(Ch, a, b) / w), 100 * (1 - seg(Cl, a, b) / w)))
for name in sys.argv[4:]:
    r = T[name]; tool = r['tool']; i = int(name[-2:]); rng = PL.rng_of('loopr', tool, i); a = str(rng.choice(PL.NAMES[r['split']])); x = PL.dry(a); p = R.ZOO[tool][2](rng); L = len(x)
    print('%s (direct over reverb %.1f dB, clip of %.1f s)' % (name, r['drr'], L / SR), flush=True)
    # six times over
    wet = sf.read(r['path'], dtype='float64', always_2d=True)[0]; direct = sf.read(r['truth'], dtype='float64', always_2d=True)[0]; p1, p2 = render(TAG + '_' + name + '_long', r['path'])
    table('six times over', wet, direct, p1, p2, [(k * L, (k + 1) * L, 'time %d through' % (k + 1)) for k in range(PL.TIMES)] + [(PL.TIMES * L, len(wet), 'the ring-out')])
    # once
    w = R.apply(tool, x, p, 3.0); pk = np.abs(w).max(); g = 0.98 / pk if pk >= 0.98 else 1.0; n = len(w)
    cl = R.impulse(SR * 6, CLICK); cl[:, 1] = 0; hl = R.apply(tool, cl, p)[CLICK:]; cr = R.impulse(SR * 6, CLICK); cr[:, 0] = 0; hr = R.apply(tool, cr, p)[CLICK:]
    e = (hl ** 2).sum(1) + (hr ** 2).sum(1); k = int(np.argmax(e)); a0, b0 = max(0, k - 48), k + 120
    d1 = np.stack([ss.fftconvolve(x[:, 0], hl[a0:b0, o])[:n] + ss.fftconvolve(x[:, 1], hr[a0:b0, o])[:n] for o in range(2)], 1)
    if a0 > 0: d1 = np.vstack([np.zeros((a0, 2)), d1])[:n]
    if len(d1) < n: d1 = np.vstack([d1, np.zeros((n - len(d1), 2))])
    pi = D + 'wav/long_once/' + name + '.wav'
    if not os.path.exists(pi): sf.write(pi, w * g, SR, subtype='PCM_16')
    wet1 = sf.read(pi, dtype='float64', always_2d=True)[0]; q1, q2 = render(TAG + '_' + name + '_once', pi)
    table('once', wet1, d1 * g, q1, q2, [(0, L, 'the clip'), (L, n, 'the ring-out')])
