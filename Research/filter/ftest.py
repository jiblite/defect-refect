# The fixed-filter reader on the scoreboard: the curve it reads on the 8 filter cases against the true one, and how sure it
# is that a filter is there on every scoreboard sound and on the clean corpus.
import sys, numpy as np, torch
from common import *
import ffeat, ftrain
def load_net(path, ch=32):
    net = ftrain.Net(ch); net.load_state_dict(torch.load(path)); net.eval(); return net
def read(net, E, rounds=1):
    """E (frames, bands) -> (curve dB per band, sureness logit). rounds > 1: the read curve is taken off and the rest read again."""
    tot = np.zeros(E.shape[1]); e = E.astype(np.float64); first = None
    for r in range(rounds):
        X, ref = (ffeat.curves6 if net.c[0].in_channels == 7 else ffeat.curves)(e)
        with torch.no_grad(): c, t = net(torch.tensor(X[None].astype(np.float32)))
        c = c[0].numpy().astype(np.float64); first = float(t[0]) if first is None else first; tot += c; e = e * 10 ** (-c / 10)
    return tot, first
def true_curve(c):
    x, y = c['dry'], c['wet']; Ex = ffeat.energies(x); Ey = ffeat.energies(y); return 10 * np.log10(Ey.mean(0) + 1e-30) - 10 * np.log10(Ex.mean(0) + 1e-30), 10 * np.log10(Ex.mean(0) + 1e-30)
if __name__ == '__main__':
    net = load_net(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32); Z = np.load('data/cases_E.npz'); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    pts = [100, 200, 400, 800, 1600, 3200, 6400, 12800]; ib = [int(np.argmin(np.abs(ffeat.FC - p))) for p in pts]
    for d in cases('filter'):
        for w in 'AB':
            c = truth.load(d, w); T, Sx = true_curve(c); T = T - np.max(T[Sx > Sx.max() - 40]) if False else T; E = Z['filter|%s|%s' % (c['name'], w)]; L, t = read(net, E, rounds)
            aud = Sx > Sx.max() - 50; off = T - L; off = off - np.median(off[aud & (np.abs(T) < 3)]) if (aud & (np.abs(T) < 3)).any() else off
            print('%-17s %s sure %+5.1f | true  %s | read  %s | off by %.1f dB rms where the clean sound is heard' % (c['name'], w, t, ' '.join('%+4.0f' % T[i] for i in ib), ' '.join('%+4.0f' % L[i] for i in ib), np.sqrt((np.clip(off[aud], -30, 30) ** 2).mean())), flush=True)
    fams = {}
    for k in Z.files:
        fam, name, w = k.split('|'); L, t = read(net, Z[k]); fams.setdefault(fam, []).append(t)
    C = np.load('data/corpus_E.npz'); clean = np.array([read(net, e)[1] for e in C['E']])
    for thr in (0.0, 1.0, 2.0, 3.0):
        print('sure above %+.0f: clean corpus %4.1f %% |' % (thr, 100 * (clean > thr).mean()), ' '.join('%s %d/%d' % (f[:5], (np.array(v) > thr).sum(), len(v)) for f, v in sorted(fams.items())))
