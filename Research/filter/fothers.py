# The filters shown on the scoreboard's sounds of the other effects: which of them are really there? Per sound that the
# engine would show (at the default, or at all), the cut the effect truly made (the sound after it against the sound
# before it, band by band, where the sound before it can be heard) beside the cut that was read.
#   python3 fothers.py net.pt [net.pt ...]
import collections
from common import *
import ffeat, fengine
from fgrid import verdict
SURE_AT = 2.5; LOW_AT = 0.0; DEEP_AT = 6.0
if __name__ == '__main__':
    nets = sys.argv[1:]; Z = np.load('data/cases_E.npz'); rows = []
    for k in Z.files:
        fam, name, w = k.split('|')
        if fam == 'filter': continue
        Ew = Z[k].astype(np.float64); F = fengine.find(None, nets, E=Ew); depth, kd, apart = verdict(F, Ew)
        if not (F['sure'] >= LOW_AT and depth >= DEEP_AT and kd > 0): continue
        c = truth.load(SETS + fam + '/' + name + '/', w); Ed = ffeat.energies(c['dry']); T = 10 * np.log10(Ew.mean(0) + 1e-30) - 10 * np.log10(Ed.mean(0) + 1e-30); Sx = 10 * np.log10(Ed.mean(0) + 1e-30); aud = Sx > Sx.max() - 50
        ends = max(-float(np.median(T[:6])), -float(np.median(T[-6:])))            # the cut at either end of the range, over half an octave
        rows.append((fam, name, w, F['sure'], kd, depth, max(0.0, float(np.max(-T[aud]))), ends)); print('%-11s %-22s %s sure %4.1f kind %d | read a cut of %2.0f dB | truly cut by %2.0f dB at the most, %3.0f dB at an end' % rows[-1], flush=True)
    for tag, at in (('at the default', SURE_AT), ('at all', LOW_AT)):
        r = [x for x in rows if x[3] >= at]; real = [x for x in r if x[7] >= 6.0]
        print('shown %s: %d | truly cut by 6 dB or more at an end: %d (%s) | not: %d (%s)' % (tag, len(r), len(real), ' '.join('%s %d' % kv for kv in sorted(collections.Counter(x[0] for x in real).items())),
              len(r) - len(real), ' '.join('%s %d' % kv for kv in sorted(collections.Counter(x[0] for x in r if x[7] < 6.0).items()))))
