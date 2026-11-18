# How loud each voice of a chorus is, per frequency range, from the wet sound alone.
# The dry sound has no copy of itself along the voices' delay curves: the voices' loudness is right when the result of
# taking them out no longer matches itself read along those curves. So: take them out with the loudness so far, measure
# what is left of each voice in each range (how much of the result is explained by the result read along the curve),
# add that, and again.
#   bands(y)                 the sound in NB frequency ranges that add up to the sound again
#   left(x, D)               per range and voice: the part of x that is x read along the voice's curve  (NB, V)
#   remove(y, D, G)          the voices taken out, G (NB, V) their loudness per range
#   find(y, D)               G for a wet sound
#   python3 gains.py case[:A|B] ...
import sys, numpy as np, numba
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
import chorus as C, inv, prim
from prim import bands, NB, EDGES
SR = 48000
MOST = 0.9                                                    # the voices' loudness in one range may add up to this at most (above 1 the taking out runs away)

@numba.njit(cache=True)
def _along(x, d):
    """x read along a delay curve: x(n - d[n])."""
    n = len(x); r = np.zeros(n)
    for i in range(n):
        p = i - d[i]
        if p >= 1.0: r[i] = inv._lag4(x, p)
    return r

_BIN = None
def _bin_band():
    global _BIN
    if _BIN is None: f = np.arange(prim.SN // 2 + 1) * SR / prim.SN; _BIN = np.searchsorted(EDGES, f, side='right')
    return _BIN

def left(x, D):
    """(NB, V): per range, how much of each voice is in x. Loud and quiet stretches and frequencies count alike: the products
    are divided by the sound's power there, smoothed widely (770 Hz, 0.2 s; a narrow smoothing follows the copy's own comb
    and reads too low)."""
    X = prim.spec(x); P = np.abs(X) ** 2; Ps = prim.smooth2(P, 33, 9); w = 1.0 / (Ps + 1e-4 * P.mean() + 1e-30); w = w * (Ps > Ps.max() * 1e-7)
    R = [prim.spec(_along(x, D[v])) for v in range(len(D))]; V = len(R); bb = _bin_band(); out = np.zeros((NB, V))
    for b in range(NB):
        m = bb == b; A = np.zeros((V, V)); rhs = np.zeros(V)
        for v in range(V):
            rhs[v] = (w[m] * (X[m] * np.conj(R[v][m])).real).sum()
            for u in range(v, V): A[v, u] = A[u, v] = (w[m] * (R[v][m] * np.conj(R[u][m])).real).sum()
        out[b] = np.linalg.solve(A + (1e-3 * np.trace(A) / max(V, 1) + 1e-30) * np.eye(V), rhs)
    return out

def remove(y, D, G, B=None):
    B = bands(y) if B is None else B; return sum(inv._inverse(B[b], D, np.ascontiguousarray(G[b])) for b in range(NB))

def put(x, D, G, s=1.0):
    B = bands(x); return x + s * sum(inv._forward(B[b], D, np.ascontiguousarray(G[b])) - B[b] for b in range(NB))

def capped(G):
    s = np.abs(G).sum(1, keepdims=True); return G * np.minimum(1.0, MOST / np.maximum(s, 1e-9))

def find(y, D, rounds=6, say=None):
    B = bands(y); G = np.zeros((NB, len(D)))
    for r in range(rounds):
        x = remove(y, D, G, B); L = left(x, D); G = capped(G + L)
        if say: say(r, G, L)
    return G

if __name__ == '__main__':
    from lab import load, SETS, score
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for a in sys.argv[1:]:
        name, which = (a.split(':') + ['A'])[:2]; c = load(SETS + 'chorus/' + name, which); n = len(c['wet']); print(name, which, c['meta']['params'], flush=True)
        out = []
        for ch in range(2):
            A, act = C.picture(c['wet'][:, ch]); P = C.peaks(A, act); V = C.voices(P, len(A)); D = np.stack([inv.delays(v, n) for v in V])
            for v in V: print('  ch%d %s %.4f Hz  %.3f +- %.3f ms  on %.2f' % (ch, 'tri' if v['kind'] else 'sin', v['rate'], v['mid'] / C.FS * 1e3, abs(v['half']) / C.FS * 1e3, v['on']), flush=True)
            G = find(c['wet'][:, ch], D, say=lambda r, G, L: print('    round %d  left %s' % (r, ' | '.join(' '.join('%5.2f' % x for x in L[:, v]) for v in range(L.shape[1])))))
            for v in range(len(V)): print('  ch%d voice %d loudness per range: %s' % (ch, v, ' '.join('%.2f' % g for g in G[:, v])))
            print('  ch%d on the DRY sound: %s' % (ch, ' | '.join(' '.join('%5.2f' % x for x in left(c['dry'][:, ch], D)[:, v]) for v in range(len(V)))), flush=True)
            out.append(remove(c['wet'][:, ch], D, G))
        print('  taken out: %.1f' % score.removed(c['dry'], c['wet'], np.stack(out, 1))[0], flush=True)
