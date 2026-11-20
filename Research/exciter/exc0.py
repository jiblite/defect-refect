# The exciter study, first look: can the highs of a sound be foretold from its upper middle bent (squared, cubed)? Fitted on
# half of the sound (every other quarter second), tested on the other half: the share of the highs' power foretold there.
from common import *
def bp(x, lo, hi=None, order=6):
    sos = ss.butter(order, lo / (SR / 2), 'high', output='sos') if hi is None else ss.butter(order, [lo / (SR / 2), min(hi, 0.49 * SR) / (SR / 2)], 'band', output='sos'); return ss.sosfiltfilt(sos, x)
def lagged(r, taps):
    n = len(r); return np.stack([np.roll(r, k) for k in range(-taps // 2, taps - taps // 2)], 1)
def coupling(y, fs, taps=24):
    """Cross-validated share of the band above 2 fs foretold from the square and the cube of the band fs .. 2 fs (and of all above fs)."""
    b = bp(y, fs, 2 * fs); a = bp(y, fs); h = bp(y, 2 * fs); n = len(y); blk = (np.arange(n) // (SR // 4)) % 2
    regs = [bp(b * b, 2 * fs), bp(b ** 3, 2 * fs), bp(b * np.abs(b), 2 * fs)]
    X = np.concatenate([lagged(r, taps) for r in regs], 1); tr = blk == 0; te = blk == 1
    A = X[tr]; w = np.linalg.lstsq(A.T @ A + 1e-6 * np.trace(A.T @ A) / A.shape[1] * np.eye(A.shape[1]), A.T @ h[tr], rcond=None)[0]; pred = X[te] @ w
    return max(-1.0, 1.0 - ((h[te] - pred) ** 2).sum() / max((h[te] ** 2).sum(), 1e-30)), 10 * np.log10(max((h ** 2).mean(), 1e-30) / max((y ** 2).mean(), 1e-30))
for fam in sys.argv[1:] or ('exciter',):
    for d in cases(fam):
        for w in 'A':
            c = truth.load(d, w); row = '%-18s %s %-4s' % (c['name'], w, kind(c['meta'][w]))
            for label, v in (('dry', c['dry']), ('wet', c['wet'])):
                m = v.mean(1); row += ' | %s:' % label
                for fs in (1500, 2500, 4000, 6000):
                    ev, lev = coupling(m, fs); row += ' %5.1f%% (%3.0f)' % (100 * ev, lev)
            print(row, flush=True)
