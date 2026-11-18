# Which is louder, the sound or its copy? Two ways to tell, tried on sounds where it is known:
#  (a) both ways of taking the copy out (usual, backwards) and how peaked / how free of the copy each result is
#  (b) from the wet sound alone: at a sharp event the sound comes first and the copy after it; if the later one is the
#      stronger, the copy is louder. Per range, with e = |y| and ed = |y read along the curve|: A = sum e^2 ed, B = sum e ed^2;
#      log(A / B) over 0 says the copy is louder.
#   python3 t26.py cho_tap_00:A x:x02_A ...
import sys, json, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
import chorus as C, engine as E, gains as Gn, prim
from lab import load, SETS, score
def sound(a):
    k, name = a.split(':', 1)
    if k == 'x': return sf.read('/dev/shm/chx/%s_wet.wav' % name, dtype='float64', always_2d=True)[0], sf.read('/dev/shm/chx/%s_dry.wav' % name, dtype='float64', always_2d=True)[0]
    c = load(SETS + 'chorus/' + k, name); return c['wet'], c['dry']
def lean(y1, D):
    B = prim.bands(y1); out = []
    for b in range(Gn.NB):
        e = np.abs(B[b]); ed = np.abs(Gn._along(B[b], D)); A = float((e * e * ed).sum()); Bq = float((e * ed * ed).sum()); out.append(np.log(max(A, 1e-300) / max(Bq, 1e-300)))
    return out
def tsparse(x):
    """How unevenly the sound's strength is spread in time, per range (mean of the fourth power over the square of the mean square of the range's samples), averaged over the ranges 2 .. 7."""
    B = prim.bands(x); return float(np.mean([(B[b] ** 4).mean() / max((B[b] ** 2).mean() ** 2, 1e-300) for b in range(2, Gn.NB)]))
import os; os.makedirs('/dev/shm/chw', exist_ok=True)
for a in sys.argv[1:]:
    wet, dry = sound(a); n = len(wet); outs = {'usual': wet.copy(), 'other': wet.copy()}
    for ch in range(1 if np.array_equal(wet[:, 0], wet[:, 1]) else 2):
        A, act = C.picture(wet[:, ch]); T = len(A); P = C.peaks(A, act); del A; V = C.voices(P, T, need=E.NEED)
        if not V: print('%s side %d: no voices' % (a, ch)); continue
        V = sorted(V, key=lambda v: -v['on'])[:1]; v = V[0]; D = E.curves(V, n); G, xu = E.loudness(wet[:, ch], V, D); H, xo, U = E.backwards(wet[:, ch], V)
        h0, a0 = C.ride(wet[:, ch], v); hu, _ = C.ride(xu, v); ho, _ = C.ride(xo, v); L = lean(wet[:, ch], D[0])
        print('%s side %d on %.2f | g %.2f h %.2f | along %.3f -> usual %.3f other %.3f | peaked wet %.1f usual %.1f other %.1f (dry %.1f) | uneven wet %.1f usual %.1f other %.1f (dry %.1f) | lean %s  sum %.2f' % (a, ch, v['on'], G[3:6, 0].mean(), H[3:6, 0].mean(), h0[a0].mean(), hu[a0].mean(), ho[a0].mean(), E.spiky(wet[:, ch]), E.spiky(xu), E.spiky(xo), E.spiky(dry[:, ch]), tsparse(wet[:, ch]), tsparse(xu), tsparse(xo), tsparse(dry[:, ch]), ' '.join('%+.2f' % l for l in L), float(np.sum(L[2:]))), flush=True)
        outs['usual'][:, ch] = xu; outs['other'][:, ch] = xo; np.savez('/dev/shm/chw/%s_%d.npz' % (a.replace(':', '_'), ch), wet=wet[:, ch], dry=dry[:, ch], xu=xu, xo=xo, D=D[0], G=G[:, 0], H=H[:, 0], along=np.array([h0[a0].mean(), hu[a0].mean(), ho[a0].mean()]))
        if ch == 0 and np.array_equal(wet[:, 0], wet[:, 1]): outs['usual'][:, 1] = xu; outs['other'][:, 1] = xo
    print('%s score: usual %.1f   other %.1f' % (a, score.removed(dry, wet, outs['usual'])[0], score.removed(dry, wet, outs['other'])[0]), flush=True)
