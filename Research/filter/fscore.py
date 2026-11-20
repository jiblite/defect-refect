# The fixed-filter engine's prototype on the 8 scoreboard cases: found or not, out and onto scores as the scoreboard's.
import sys
from common import *
import fengine
path = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1; tot = []
for d in cases('filter'):
    A = truth.load(d, 'A'); B = truth.load(d, 'B'); F = fengine.find(A['wet'], path, rounds)
    o40 = score.removed(A['dry'], A['wet'], fengine.own(A['wet'], F, 0.0))[0]; of = score.removed(A['dry'], A['wet'], fengine.own(A['wet'], F, 0.0, by_floor=True))[0]
    o24 = score.removed(A['dry'], A['wet'], fengine.own(A['wet'], F, 0.0, boost=24.0))[0]; half = score.removed(A['dry'], A['wet'], fengine.own(A['wet'], F, 0.5))[0]
    on = score.removed(B['wet'], B['dry'], fengine.onto(B['dry'], F, 1.0))[0]; tot.append((o40, on))
    print('%-17s sure %+5.1f | out %4.0f (lift up to 24 dB: %4.0f, lift held by the floor: %4.0f, half way: %4.0f) | onto %4.0f' % (A['name'], F['sure'], o40, o24, of, half, on), flush=True)
print('mean out %.0f onto %.0f | median out %.0f onto %.0f' % (np.mean([t[0] for t in tot]), np.mean([t[1] for t in tot]), np.median([t[0] for t in tot]), np.median([t[1] for t in tot])))
