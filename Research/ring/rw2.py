# The level window on cases of rk2.py (a tone put on a clip of the corpus), by their numbers: eight cycles (a power of
# two, 21 ms at least) against whole cycles (two boxes of LEAST samples at least each).   python3 rw2.py k [k ...]
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import ring as E, rf1, rk2
old_window = E.window
def whole(least):
    def f_(f): P = SR / f; k = int(np.ceil(least / P)); return 2 * int(round(k * P))
    return f_
C = rk2.cases()
for k in [int(v) for v in sys.argv[1:]]:
    clip, f, rho, ph = C[k]; x = truth.rd('/tmp/claude-0/v2/corpus/dry/' + clip + '.wav'); y = rf1.plain(x, f, rho, ph)
    print('%d %s true f %.4f rho %.3f' % (k, clip if not clip.startswith('real_') else 'real_XX', f, rho))
    for name, win in (('8 cycles', old_window), ('whole, 480 at least', whole(480)), ('whole, 240 at least', whole(240)), ('whole, 960 at least', whole(960))):
        E.window = win; t0 = time.time(); r = E.find(y); dt = time.time() - t0
        if 'f' in r:
            left = 0.0
            if r['found']: kk = float((r['x'] * x).sum() / (x ** 2).sum()); left = db(r['x'] - kk * x, kk * x)
            print('    %-22s W %5d found %d f %.5f rho %.4f floor %.1e win %.3f match %.2f dip %.1f | left %6.1f dB | %d passes %.1f s' % (name, win(r['f']), r['found'], r['f'], r['rho'], r['eps'], r['win'], r['match'], r['dip'], left, r['passes'], dt), flush=True)
        else: print('    %-22s not measured (first win %.3f)' % (name, r.get('first_win', -9)), flush=True)
    E.window = old_window
