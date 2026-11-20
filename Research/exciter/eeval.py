# The exciter study: the reader of added highs on the 20 outside exciter cases (their truth is known), on the same sounds
# before the exciter, on every other scoreboard sound and on the clean corpus.
#   python3 eeval.py net.pt
import sys, numpy as np, torch
from common import *
import ffeat, ftest
net = ftest.load_net(sys.argv[1], 48); FC = ffeat.FC
roots = ['/tmp/claude-0/v2/bench/sets/exciter/', '/tmp/claude-0/v2/four/sets2/exciter/']; pts = [2000, 4000, 8000, 12000, 16000]; ib = [int(np.argmin(np.abs(FC - p))) for p in pts]
wet, dry, offs, sizes = [], [], [], []
for r in roots:
    for d in sorted(glob.glob(r + '*/')):
        for w in 'AB':
            c = truth.load(d, w); Ed = ffeat.energies(c['dry']); Ew = ffeat.energies(c['wet']); n = min(len(Ed), len(Ew)); Ed, Ew = Ed[:n], Ew[:n]
            mw = 10 * np.log10(Ew.mean(0) + 1e-30); T = np.clip(mw - 10 * np.log10(Ed.mean(0) + 1e-30), -40, 40); aud = mw > mw.max() - 60
            L, t = ftest.read(net, Ew); Ld, td = ftest.read(net, Ed); off = float(np.sqrt(np.mean((T - L)[aud] ** 2))); size = float(np.sqrt(np.mean(T[aud] ** 2)))
            wet.append(t); dry.append(td); offs.append(off); sizes.append(size)
            print('%-5s %-20s %s sure %+5.1f (before the exciter %+5.1f) | true lift %s | read %s | off by %4.1f dB rms (doing nothing: %4.1f)' % ('new' if 'sets2' in r else 'board', c['name'], w, t, td,
                  ' '.join('%+4.0f' % T[i] for i in ib), ' '.join('%+4.0f' % L[i] for i in ib), off, size))
wet, dry = np.array(wet), np.array(dry)
for thr in (0.0, 1.0, 2.0, 3.0): print('above %.0f: %2d of %d excited sounds, %2d of %d of the same sounds before the exciter' % (thr, (wet > thr).sum(), len(wet), (dry > thr).sum(), len(dry)))
print('curve off by %.1f dB rms on average; reading nothing would be off by %.1f; better than nothing on %d of %d' % (np.mean(offs), np.mean(sizes), (np.array(offs) < np.array(sizes)).sum(), len(offs)))
Z = np.load('data/cases_E.npz'); fams = {}
for k in Z.files:
    fam, name, w = k.split('|'); fams.setdefault(fam, []).append(ftest.read(net, Z[k])[1])
C = np.load('data/corpus_E.npz'); clean = np.array([ftest.read(net, e)[1] for e in C['E']])
for thr in (0.0, 1.0, 2.0, 3.0):
    print('above %.0f: clean corpus %.1f %% |' % (thr, 100 * (clean > thr).mean()), ' '.join('%s %d/%d' % (f, (np.array(v) > thr).sum(), len(v)) for f, v in sorted(fams.items())))
