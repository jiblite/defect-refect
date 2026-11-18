# The extra chorus cases (mkx.py) through the prototype: taken out (against the dry sound) and put onto the other sound (against its wet sound).
#   python3 xbench.py [case ...]
import sys, json, time, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
import engine as E
from lab import score
D = '/dev/shm/chx/'; cases = json.load(open(D + 'cases.json')); rd = lambda p: sf.read(p, dtype='float64')[0]
for name in sys.argv[1:] or sorted(cases):
    for w, o in (('A', 'B'), ('B', 'A')):
        wet = rd(D + '%s_%s_wet.wav' % (name, w)); dry = rd(D + '%s_%s_dry.wav' % (name, w)); owet = rd(D + '%s_%s_wet.wav' % (name, o)); odry = rd(D + '%s_%s_dry.wav' % (name, o))
        F = E.find(wet); out = E.apply(wet, F, -1.0); onto = E.apply(odry, F, 1.0, own=False)
        print('%s %s %s | found %d sure %.2f | out %6.1f  onto %6.1f' % (name, w, ' '.join('%s %g' % (k[:4], v) for k, v in cases[name]['params'].items()), F['found'], F['sure'], score.removed(dry, wet, out)[0], score.removed(owet, odry, onto)[0]), flush=True); E.show(F)
