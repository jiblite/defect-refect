# The ring-mod prototype on the 8 scoreboard cases: what it measures against what the effect did, removal and onto scores.
import sys, time; sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import ring as E
def truth_gain(dry, wet):
    """The gain wet / dry fitted with the dry sound known (as rm0.py): f, a, b, phase."""
    x, y = dry.sum(1), wet.sum(1); n = len(x); t = np.arange(n) / SR; ok = np.abs(x) > 0.05 * np.abs(x).max(); g = y[ok] / x[ok]; tt = t[ok]
    G = np.zeros(n); G[ok] = g - g.mean(); F = np.abs(np.fft.rfft(G * np.hanning(n), 4 * n)); fr = np.fft.rfftfreq(4 * n, 1 / SR); f0 = fr[int(np.argmax(F[20:]) + 20)]
    for it in range(4):
        fs = f0 + np.linspace(-1, 1, 41) * (0.05 / 3 ** it); errs = []
        for f in fs: A = np.stack([np.ones_like(tt), np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)], 1); co = np.linalg.lstsq(A, g, rcond=None)[0]; errs.append(((A @ co - g) ** 2).sum())
        f0 = fs[int(np.argmin(errs))]
    A = np.stack([np.ones_like(tt), np.sin(2 * np.pi * f0 * tt), np.cos(2 * np.pi * f0 * tt)], 1); co = np.linalg.lstsq(A, g, rcond=None)[0]
    return f0, co[0], float(np.hypot(co[1], co[2])), float(np.arctan2(co[2], co[1]))
for d in cases('ringmod'):
    c = truth.load(d, 'A'); cb = truth.load(d, 'B'); dry, wet = c['dry'], c['wet']; f0, a, b, ph0 = truth_gain(dry, wet); t0 = time.time(); f = E.find(wet); dt = time.time() - t0
    line = '%-12s truth f %9.4f Hz a/b %.4f phase %+.4f | ' % (c['name'], f0, a / b, ph0)
    if 'f' in f:
        dph = np.angle(np.exp(1j * (f['phase'] - ph0))); 
        line += 'found %s f %9.4f (%+.5f) a/b %.4f phase err %+.5f (or pi %+.5f) eps %.1e win %.2f match %.2f dB dip %.1f dB offset %.1f dB | %.1f s' % (f['found'], f['f'], f['f'] - f0, f['rho'], dph, np.angle(np.exp(1j * (dph - np.pi))), f['eps'], f['win'], f['match'], f['dip'], f['offset'], dt)
        if f['found']:
            o0 = E.own(wet, f, 0.0); oh = E.own(wet, f, 0.5); o1 = E.own(wet, f, 1.0); on = E.onto(cb['dry'], f, 1.0)
            line += '\n      out %4.0f /%4.0f  half way %4.0f  untouched at the middle: %s | onto %4.0f /%4.0f' % (score.removed(dry, wet, o0)[0], removed4(dry, wet, o0)[0], score.removed(dry, wet, oh)[0], 'yes' if np.abs(o1 - wet).max() == 0 else 'NO', score.removed(cb['wet'], cb['dry'], on)[0], removed4(cb['wet'], cb['dry'], on)[0])
    print(line, flush=True)
