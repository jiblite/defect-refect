# The other way of taking a copy out (engine.backwards) against the usual one, on sounds whose copy is louder than the sound.
#   python3 t25.py cho_pedalboard_00:A x:x02_A ...
import sys, json, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
import chorus as C, engine as E, gains as Gn
from lab import load, SETS, score
def sound(a):
    k, name = a.split(':', 1)
    if k == 'x': return sf.read('/dev/shm/chx/%s_wet.wav' % name, dtype='float64', always_2d=True)[0], sf.read('/dev/shm/chx/%s_dry.wav' % name, dtype='float64', always_2d=True)[0]
    c = load(SETS + 'chorus/' + k, name); return c['wet'], c['dry']
for a in sys.argv[1:]:
    wet, dry = sound(a); n = len(wet); outs = {'usual': wet.copy(), 'other': wet.copy()}; print(a, flush=True)
    for ch in range(2):
        A, act = C.picture(wet[:, ch]); T = len(A); P = C.peaks(A, act); del A; V = C.voices(P, T, need=E.NEED)
        if not V: print('  side %d: no voices' % ch); continue
        V = sorted(V, key=lambda v: -v['on'])[:1]; v = V[0]; D = E.curves(V, n); G, xu = E.loudness(wet[:, ch], V, D); H, xo, U = E.backwards(wet[:, ch], V)
        h0, a0 = C.ride(wet[:, ch], v); hu, _ = C.ride(xu, v); ho, _ = C.ride(xo, v)
        print('  side %d %s %.4f Hz %.2f+-%.2f ms on %.2f | usual: g %s  along %.3f -> %.3f  peaked %.1f -> %.1f | other: h %s  along -> %.3f  peaked -> %.1f | the dry sound: peaked %.1f' % (ch, 'tri' if v['kind'] else 'sin', v['rate'], v['mid'] / C.FS * 1e3, abs(v['half']) / C.FS * 1e3, v['on'], ' '.join('%.2f' % g for g in G[:, 0]), h0[a0].mean(), hu[a0].mean(), E.spiky(wet[:, ch]), E.spiky(xu), ' '.join('%.2f' % g for g in H[:, 0]), ho[a0].mean(), E.spiky(xo), E.spiky(dry[:, ch])), flush=True)
        outs['usual'][:, ch] = xu; outs['other'][:, ch] = xo
    print('  score: usual %.1f   other %.1f' % (score.removed(dry, wet, outs['usual'])[0], score.removed(dry, wet, outs['other'])[0]), flush=True)
