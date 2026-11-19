# The crush engine (crush.py) on the scoreboard's bitcrush cases: what is found against what was set, and the crusher
# put onto sound B against B through the real effect.   python3 bc2.py
import sys, os, json, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import crush as E
for d in cases('bitcrush'):
    m = json.load(open(d + 'meta.json')); A = truth.load(d, 'A'); B = truth.load(d, 'B'); t0 = time.time(); f = E.find(A['wet']); dt = time.time() - t0
    p = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m['params'].items() if k in ('bits', 'samples', 'down', 'morph', 'anti_aliasing', 'quant', 'rate', 'post_filt')}
    g = f.get('grids', [None])[0]; h = f.get('hold', {})
    print('%-18s %s' % (os.path.basename(d[:-1]), p)); print('     found %d | grid: %s | hold: P %.5f c %.3f share %.4f (%d changes) | %.2f s' % (f['found'], 'none' if g is None else 'step %.7f (1/%.3f, %.2f bits) offset %+.3f steps share %.4f levels %d' % (g['step'], 1 / g['step'], g['bits'], g['offset'] / g['step'], g['share'], g['levels']), h.get('P', 0), h.get('c', 0), h.get('share', 0), h.get('changes', 0), dt))
    if f['found']:
        o = E.onto(B['dry'], f); n = min(len(o), len(B['wet'])); print('     onto B: %.1f of 100 (B through the real effect), waveform %.1f dB beside it' % (score.removed(B['wet'][:n], B['dry'][:n], o[:n])[0], db(o[:n] - B['wet'][:n], B['wet'][:n])))
