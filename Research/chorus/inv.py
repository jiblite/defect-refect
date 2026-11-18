# The chorus taken out and put on: with the voices' delay curves known, the wet sound is  y[n] = x[n] + sum_v g_v x(n - d_v[n]).
#   forward(x, D, g)         puts the voices on a sound
#   inverse(y, D, g)         takes them out again: x[n] = y[n] - sum_v g_v x(n - d_v[n])  (the sound's own earlier output)
#   delays(v, n)             a voice's delay in samples at 48 kHz for every sample
#   python3 inv.py case[:A|B] ...     the voices read off the wet sound, then the score for a row of gains (same gain for all voices)
import sys, numpy as np, scipy.signal as ss, numba
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
import chorus as C
SR = 48000

def delays(v, n):
    t = np.arange(n) / (C.HOP * C.DEC); d, _ = C.curve(v, t); return d * C.DEC

@numba.njit(cache=True)
def _lag4(x, p):
    """x at the place p (not whole), four points."""
    i = int(np.floor(p)); f = p - i
    if i < 1 or i + 2 >= len(x): return 0.0
    a, b, c, d = x[i - 1], x[i], x[i + 1], x[i + 2]
    return b + 0.5 * f * (c - a + f * (2.0 * a - 5.0 * b + 4.0 * c - d + f * (3.0 * (b - c) + d - a)))

@numba.njit(cache=True)
def _inverse(y, D, g):
    n = len(y); x = np.zeros(n); V = len(g)
    for i in range(n):
        s = y[i]
        for v in range(V):
            p = i - D[v, i]
            if p >= 1.0: s -= g[v] * _lag4(x, p)
        x[i] = s
    return x

@numba.njit(cache=True)
def _forward(x, D, g):
    n = len(x); y = x.copy(); V = len(g)
    for i in range(n):
        for v in range(V):
            p = i - D[v, i]
            if p >= 1.0: y[i] += g[v] * _lag4(x, p)
    return y

def split(y, fc):
    """The sound below fc and the rest (no shift in time)."""
    if fc is None or fc >= SR / 2: return y, np.zeros_like(y)
    sos = ss.butter(4, fc, 'low', fs=SR, output='sos'); lo = ss.sosfiltfilt(sos, y); return lo, y - lo

def inverse(y, D, g, fc=None):
    """One channel. Below fc the voices are taken out sample by sample; above it (where the delay is not known finely enough) only the level is set right."""
    g = np.asarray(g, float); lo, hi = split(y, fc); return _inverse(lo, D, g) + hi / np.sqrt(1.0 + (g ** 2).sum())

def forward(x, D, g):
    return _forward(x, D, np.asarray(g, float))

if __name__ == '__main__':
    from lab import load, SETS, score
    import os
    for a in sys.argv[1:]:
        name, which = (a.split(':') + ['A'])[:2]; c = load(SETS + 'chorus/' + name, which); n = len(c['wet']); print(name, which, c['meta']['params'], flush=True)
        Vs = []
        for ch in range(2):
            A, act = C.picture(c['wet'][:, ch]); P = C.peaks(A, act); V = C.voices(P, len(A)); Vs.append(V)
            for v in V: print('  ch%d %s %.4f Hz  %.3f +- %.3f ms  phase %.3f  on %.2f height %.3f' % (ch, 'tri' if v['kind'] else 'sin', v['rate'], v['mid'] / C.FS * 1e3, abs(v['half']) / C.FS * 1e3, v['phase'], v['on'], v['height']), flush=True)
        Ds = [np.stack([delays(v, n) for v in V]) if V else np.zeros((0, n)) for V in Vs]
        for fc in (None, 6000.0, 3000.0):
            row = []
            for g in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                out = np.stack([inverse(c['wet'][:, ch], Ds[ch], [g] * len(Ds[ch]), fc) for ch in range(2)], 1); row.append(score.removed(c['dry'], c['wet'], out)[0])
            print('  fc %s: ' % fc + ' '.join('g %.1f: %6.1f' % (g, r) for g, r in zip((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), row)), flush=True)
