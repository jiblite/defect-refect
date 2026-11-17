#!/usr/bin/env python3
"""Compares the app's verification step (Source/Core/Verify.h through Tests/verify.cpp, built as drverify) with the
Python side (Research/detect).

  python3 Tools/verify_parity.py path/to/drverify sound.wav [sound.wav ...] [--resources Resources] [--proto folder]

For each sound (48 kHz stereo WAV, any length) it checks
  the blocks the sound is judged in,
  evidence(): the 55 physical measurements of every block, number by number (evidence.py),
  the scores: recomputed here from what the C++ detectors gave per window, with the evidence measured here, the trees
  read from verify.bin and the same rule over the blocks (vref.py from_dump).
With --proto (the folder of the first prototype's feat.py, detect.py and echo.py; needs torch and numba as well) the
detectors run in Python too (appdet.py), so that nothing is taken from the C++: then the scores can differ where the
two CNN implementations differ in the last digits and a tree's threshold lies between them.
Needs numpy, scipy, scikit-learn and soundfile. Ends with status 1 when a shown / not shown decision at 65 % differs
or an evidence number is off by more than one part in a million.
"""
import sys, os, json, subprocess, argparse

ap = argparse.ArgumentParser(); ap.add_argument('drverify'); ap.add_argument('wavs', nargs='+')
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument('--resources', default=os.path.join(here, '..', 'Resources')); ap.add_argument('--proto', default='')
ap.add_argument('--detect', default=os.path.join(here, '..', 'Research', 'detect'))
A = ap.parse_args(); sys.path.insert(0, A.detect); os.environ['DR_RESOURCES'] = A.resources
if A.proto: os.environ['DR_PROTO'] = A.proto
import numpy as np, soundfile as sf
import vref, export_verify

model = export_verify.read(os.path.join(A.resources, 'verify.bin')); bad = False; worst_ev = 0.0; worst_sc = 0.0; worst_py = 0.0; flips = 0; flips_py = 0; total = 0
for path in A.wavs:
    y, sr = sf.read(path, always_2d=True, dtype='float64'); assert sr == 48000, path + ': 48 kHz expected'
    y = np.repeat(y, 2, 1) if y.shape[1] == 1 else y[:, :2]
    run = subprocess.run([A.drverify, A.resources, path], capture_output=True, text=True)
    if run.returncode: print(path, 'drverify failed:', run.stderr.strip()); bad = True; continue
    d = json.loads(run.stdout); bl = vref.blocks(len(y)); same_blocks = [(b['start'], b['len']) for b in d['blocks']] == bl
    wins = all(b['win'] == vref.block_windows([w['start'] for w in d['win']], b['start'], b['len']) for b in d['blocks'])
    final, per, evs = vref.from_dump(d, y, model)
    ev = max((abs(e[k] - b['ev'][k]) / (1e-9 + max(abs(e[k]), abs(b['ev'][k]))) for b, e in zip(d['blocks'], evs) if e is not None for k in e), default=0.0)
    sc = max(abs(final[k] - d['final'][k]) for k in final); fl = [k for k in final if (final[k] > 0.65) != (d['final'][k] > 0.65)]
    worst_ev = max(worst_ev, ev); worst_sc = max(worst_sc, sc); flips += len(fl); total += len(final)
    line = '%s: %.1f s, %d blocks (%s), %d judged; evidence off by at most %.1e (relative), scores by %.1e, decisions that differ: %s' % (
        os.path.basename(path), len(y) / 48000, len(bl), 'same' if same_blocks and wins else 'DIFFERENT', len(next(iter(d['scores'].values()), [])), ev, sc, ', '.join(fl) or 'none')
    if A.proto:
        py, _ = vref.verify_sound(y, model); s2 = max(abs(py[k] - d['final'][k]) for k in py); f2 = [k for k in py if (py[k] > 0.65) != (d['final'][k] > 0.65)]
        worst_py = max(worst_py, s2); flips_py += len(f2); line += '; all in Python: scores off by at most %.1e, decisions that differ: %s' % (s2, ', '.join(f2) or 'none')
    print(line); print('   shown at 65 %%: %s   (C++, CPU time: %.0f ms for the detectors, %.0f ms for the verification)' % (', '.join('%s %.0f %%' % (k, 100 * v) for k, v in sorted(d['final'].items()) if v > 0.65) or 'nothing', d['ms']['inspect'], d['ms']['verify']))
    bad = bad or not (same_blocks and wins) or bool(fl) or ev > 1e-6
print('all %d sounds: evidence off by at most %.1e (relative), scores by %.1e, %d of %d decisions differ' % (len(A.wavs), worst_ev, worst_sc, flips, total)
      + ('; all in Python: scores off by at most %.1e, %d decisions differ' % (worst_py, flips_py) if A.proto else ''))
sys.exit(1 if bad else 0)
