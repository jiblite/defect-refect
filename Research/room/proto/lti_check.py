# Is each outside reverb linear and time-invariant, and does its click response describe what it does to a sound?
# For one case per effect: full 2x2 click responses (click left only, right only), early click against late click, and wet against dry * response.
import sys, json, glob, os
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, soundfile as sf, scipy.signal as ss
from zoo import ZOO, apply, SR
from multiprocessing import Pool
S = '/tmp/claude-0/v2/bench/sets/reverb/'; C = '/tmp/claude-0/v2/corpus/dry/'
def rd(p): return sf.read(p, always_2d=True)[0].astype(np.float64)
def db(a, b): return 10 * np.log10((a ** 2).sum() / max((b ** 2).sum(), 1e-30) + 1e-30)
def one(d):
    m = json.load(open(d + 'meta.json')); name = os.path.basename(d.rstrip('/')); p = m['params']; eff = m['effect']
    n = SR * 8; out = []
    def click(ch, at):
        x = np.zeros((n, 2))
        if ch in (0, 2): x[at, 0] = 1.0
        if ch in (1, 2): x[at, 1] = 1.0
        return apply(eff, x, p)
    hL = click(0, 480)[480:]; hR = click(1, 480)[480:]; hB = click(2, 480)[480:]; hBl = click(2, 3 * SR)[3 * SR:]
    L = min(len(hBl), len(hB)); add = db(hB - hL - hR, hB); late = db(hB[:L] - hBl[:L], hB[:L])
    dry = rd(C + m['A'] + '.wav'); wet = rd(d + 'A_wet.flac') / m['gainA']; N = len(wet)
    pred2 = np.stack([ss.fftconvolve(dry[:, 0], hL[:, c])[:N] + ss.fftconvolve(dry[:, 1], hR[:, c])[:N] for c in range(2)], 1)
    ir = rd(d + 'ir.flac')[480:] / m['gainIR']; pred1 = np.stack([ss.fftconvolve(dry[:, c], ir[:, c])[:N] for c in range(2)], 1)
    return '%-24s additive %6.1f dB | early click = late click %6.1f dB | wet = dry * 2x2 response %6.1f dB | wet = dry * stored response %6.1f dB | dry L/R corr %.2f' % (name, add, late, db(wet - pred2, wet), db(wet - pred1, wet), np.corrcoef(dry[:, 0], dry[:, 1])[0, 1])
if __name__ == '__main__':
    ds = sorted(glob.glob(S + '*_00/')) + sorted(glob.glob(S + '*_03/'))
    with Pool(4) as p:
        for r in p.imap(one, ds): print(r, flush=True)
