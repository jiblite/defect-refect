# A bend with something behind it: the 96 sounds of the outside cases, read as they are, behind a low-pass and under a room.
# How many are still shown, and where the reading at the peak then stands against the true one.
#   python3 bunder.py data/bend_e4.bin
from common import *
import bfeat, bengine, bmeter
from beval import ROOTS, answer
def lowpass(y, fc): return ss.sosfilt(ss.butter(2, fc, 'low', fs=SR, output='sos'), y, axis=0)
def room(y, t60, under, seed=5):
    """A plain room: noise dying away by 60 dB in t60 seconds, its two sides apart, the room `under` dB under the direct sound."""
    rng = np.random.default_rng(seed); n = int(t60 * SR * 1.2); h = rng.standard_normal((n, 2)) * np.exp(-6.908 * np.arange(n)[:, None] / (t60 * SR)); h[: int(0.004 * SR)] = 0.0
    w = np.stack([ss.fftconvolve(y[:, i], h[:, i])[: len(y)] for i in (0, 1)], 1); return y + w * (rms(y) / max(rms(w), 1e-12)) * 10 ** (-under / 20)
WAYS = [('as they are', lambda y: y), ('behind a low-pass at 8 kHz', lambda y: lowpass(y, 8000.0)), ('behind a low-pass at 4 kHz', lambda y: lowpass(y, 4000.0)), ('behind a low-pass at 2 kHz', lambda y: lowpass(y, 2000.0)),
        ('behind a low-pass at 1 kHz', lambda y: lowpass(y, 1000.0)), ('under a room of 0.5 s, 18 dB under the sound', lambda y: room(y, 0.5, 18.0)), ('under a room of 0.5 s, 12 dB under', lambda y: room(y, 0.5, 12.0)),
        ('under a room of 0.5 s, 6 dB under', lambda y: room(y, 0.5, 6.0)), ('under a room of 1.5 s, 12 dB under', lambda y: room(y, 1.5, 12.0)), ('under a room of 1.5 s, 6 dB under', lambda y: room(y, 1.5, 6.0))]
if __name__ == '__main__':
    M = bmeter.model(sys.argv[1]); top = lambda S: max(bengine.tidy(S[0], bengine.SPREAD)[-1], bengine.tidy(S[1], bengine.SPREAD)[-1]); R = {w: [] for w, _ in WAYS}
    for root, tag in ROOTS:
        for fam in ('distortion', 'saturation'):
            for d in sorted(glob.glob(root + fam + '/*/')):
                for which in 'AB':
                    c = truth.load(d, which); T = top(answer(c, {}, 't'))
                    for w, f in WAYS:
                        X, vp = bfeat.features(f(np.asarray(c['wet'], float))); S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); R[w].append((fam, T, top(S), float(logit)))
    for w, _ in WAYS:
        r = R[w]; hard = [x for x in r if x[0] == 'distortion']; soft = [x for x in r if x[0] == 'saturation' and x[1] >= 6.0]; both = hard + soft
        sh = lambda v: sum(bengine.shown(x[3], x[2]) for x in v); kp = lambda v: sum(bengine.kept(x[3], x[2]) for x in v)
        print('%-46s shown at the default: %2d of %d distorted, %2d of %d saturated (6 dB and more) | at all: %2d, %2d | read at the peak against the true: %+5.1f dB on average, verdict %+5.1f' % (
            w, sh(hard), len(hard), sh(soft), len(soft), kp(hard), kp(soft), np.mean([x[2] - x[1] for x in both]), np.mean([x[3] for x in both])), flush=True)
