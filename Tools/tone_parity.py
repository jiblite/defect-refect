#!/usr/bin/env python3
"""Compares the app's tone engine (Tests/tone.cpp, built as drtone) with the Python prototype (Research/tone/tone.py).

  python3 Tools/tone_parity.py path/to/drtone a.wav b.wav [a2.wav b2.wav ...] [--detail 0.6] [--keep folder]

For each pair (48 kHz stereo WAVs; a = the sound that is analysed, b = the second sound) both sides work out
  a's long-term spectra, the reflections found on a and their curve, the Tone Match curve that gives b the tone of a
  (one curve, and one per channel), that curve as an impulse response, a with its reflections taken out, and b with
  the match put on.
Curves are compared in dB (largest difference), sounds as the level of the difference against the sound (dB).
Needs numpy, scipy and soundfile.
"""
import sys, os, subprocess, tempfile, argparse
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser(); ap.add_argument('drtone'); ap.add_argument('wavs', nargs='+'); ap.add_argument('--detail', type=float, default=0.6)
ap.add_argument('--keep', default=''); ap.add_argument('--proto', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Research', 'tone'))
A = ap.parse_args(); sys.path.insert(0, A.proto)
import tone

def rel_db(x, ref): return 10 * np.log10(max(float(((x - ref) ** 2).sum()), 1e-300) / max(float((ref ** 2).sum()), 1e-300))
def raw(p, rows): v = np.fromfile(p, np.float64); return v.reshape(rows, -1) if rows > 1 else v
worst_curve, worst_sound = 0.0, -999.0
for a_path, b_path in zip(A.wavs[0::2], A.wavs[1::2]):
    out = A.keep or tempfile.mkdtemp(); pre = os.path.join(out, os.path.splitext(os.path.basename(a_path))[0] + '__' + os.path.splitext(os.path.basename(b_path))[0])
    r = subprocess.run([A.drtone, a_path, b_path, pre, str(A.detail)], capture_output=True, text=True)
    if r.returncode: print(r.stdout, r.stderr); sys.exit(1)
    said = [l for l in r.stdout.split('\n') if l.startswith('reflections') or l.startswith('coherence')]
    a, sr = sf.read(a_path, dtype='float64', always_2d=True); b, _ = sf.read(b_path, dtype='float64', always_2d=True)
    a = a[:, :2] if a.shape[1] > 1 else np.repeat(a, 2, 1); b = b[:, :2] if b.shape[1] > 1 else np.repeat(b, 2, 1)
    S1 = tone.long_term_db(a, 'pmean', 'linked'); S2 = tone.long_term_db(a, 'pmean', 'dual')
    if S1.max() < -250 or tone.long_term_db(b).max() < -250: print('%s | %s: one of the sounds is silent (%s)' % (os.path.basename(a_path), os.path.basename(b_path), '; '.join(said))); continue
    info = []; grid, curve = tone.contour(a, info=info); ech = info[0]['echoes'] if info else []
    _, m = tone.match(a, b, A.detail); _, m2 = tone.match(a, b, A.detail, stereo='dual'); h = tone.to_ir(grid, m)
    flat = tone.apply_curve(a, grid, curve, 0.0); put = tone.put_curve(b, grid, m, 1.0); put2 = tone.put_curve(b, grid, m2, 1.0)
    cs = raw(pre + '.spec.f64', 3); d = {}
    d['spectra'] = float(np.abs(cs - np.vstack([S1[None], S2]))[:, np.vstack([S1[None], S2]).max(0) > -200].max())
    d['reflections curve'] = float(np.abs(raw(pre + '.comb.f64', 1) - curve).max())
    d['match'] = float(np.abs(raw(pre + '.match.f64', 1) - m).max()); d['match per channel'] = float(np.abs(raw(pre + '.match2.f64', 2) - m2).max())
    hc = raw(pre + '.ir.f64', 1); s = {'impulse response': rel_db(hc, h)}
    for name, ref in (('flat', flat), ('put', put), ('put2', put2)):
        y, _ = sf.read(pre + '.%s.wav' % name, dtype='float64', always_2d=True); s[name] = rel_db(y, ref) if float((ref ** 2).sum()) > 0 else -999.0
    worst_curve = max(worst_curve, max(d.values())); worst_sound = max(worst_sound, max(s.values()))
    print('%s | %s' % (os.path.basename(a_path), os.path.basename(b_path)))
    print('   prototype: %d reflections %s' % (len(ech), ' '.join('delay %.4f gain %.4f' % e for e in ech)))
    print('   app:       %s' % '; '.join(said))
    print('   curves, largest difference in dB: ' + ', '.join('%s %.2e' % kv for kv in d.items()))
    print('   sounds, difference against the sound in dB: ' + ', '.join('%s %.0f' % kv for kv in s.items()))
print('worst curve difference %.2e dB, worst sound difference %.0f dB' % (worst_curve, worst_sound))
