# The crush engine in C++ (Source/Core/Crush.h, through drcrush) against the Python prototype it was ported from
# (Research/crush/crush.py): the same verdict, the same grid and hold, the same sounds?
#   python3 Tools/crush_parity.py path/to/drcrush a.wav b.wav [a2.wav b2.wav ...]
# Every pair: the crusher is read off a and put onto b, and a itself is rendered with the faders moved. What is read
# is printed side by side (the two do not agree to the last digit: where two candidates are equally good the app
# takes the first and the prototype whichever its rounding prefers). The sounds are compared with the prototype's
# rendering of what the app read, so that they show the rendering alone. Exit code 1 if a verdict or the kind differs,
# a step differs by more than a hundred-thousandth, the hold differs, or a sound is further than -120 dB from the
# prototype's.
import sys, os, json, subprocess, tempfile, time, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'crush')]
import crush as E

def db(a, b): return 10 * np.log10(((a - b) ** 2).mean() / max((b ** 2).mean(), 1e-30) + 1e-30)
def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)

def main():
    exe = sys.argv[1]; paths = sys.argv[2:]; tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); pre = os.path.join(tmp, 'x'); bad = 0; worst = -999.0; found = 0
    for pa, pb in zip(paths[0::2], paths[1::2]):
        a = rd(pa); b = rd(pb); t0 = time.time(); r = subprocess.run([exe, pa, pb, pre], capture_output=True, text=True); t1 = time.time()
        if r.returncode: print(r.stderr); bad += 1; continue
        J = json.loads(r.stdout.strip().splitlines()[-1]); t2 = time.time(); F = E.find(a); t3 = time.time()
        same = bool(F['found']) == J['found'] and (not F['found'] or (F['kind'] == J['kind'] and abs(F['P'] - J['P']) < 1e-9 * max(F['P'], 1) and abs(F['c'] - J['c']) < 1e-6 and all(abs(u - v) <= 1e-5 * max(abs(u), 1e-12) for u, v in zip(F['step'], J['step']))))
        print('%s: app %.2f s (find %.2f s), prototype %.1f s | found %s / %s%s' % (os.path.basename(pa), t1 - t0, J['took'], t3 - t2, F['found'], J['found'], '' if same else '   DIFFERENT ANSWERS'), flush=True)
        if F['found'] and J['found']:
            print('     kind %s / %s, bits %.5f / %.5f, hold %.6f / %.6f from %.4f / %.4f, steps %s / %s' % (F['kind'], J['kind'], F['bits'], J['bits'], F['P'], J['P'], F['c'], J['c'], ' '.join('%.9g' % v for v in F['step']), ' '.join('%.9g' % v for v in J['step'])))
            if J['kind'] == 'soft' and F['kind'] == 'soft': print('     mix %.6f / %.6f, blend %.6f / %.6f, half width %.6f / %.6f' % (F['mix'], J['mix'], F['soft'], J['soft'], F['half'], J['half']))
        bad += int(not same)
        if not (F['found'] and J['found'] and same): continue
        found += 1; n = len(a); f = dict(found=True, kind=J['kind'], step=J['step'], offset=J['offset'], P=J['P'], c=J['c'], mix=J['mix'], soft=J['soft'])        # what the app read, for the prototype's rendering
        d1 = float(np.abs(np.fromfile(pre + '.same.f64').reshape(2, n).T - a).max()); bad += int(d1 != 0.0); out = ['middle %.0e' % d1]
        refs = [('onto', E.onto(b, f, 1.0)), ('onto2', E.onto(b, f, 2.0)), ('onto05', E.onto(b, f, 0.5)), ('coarse', E.onto(b, f, 1.0, 2.0, 2.0)), ('own2', E.own(a, f, 2.0)), ('owncoarse', E.own(a, f, 1.0, 2.0, 2.0))]
        if J['kind'] == 'soft': refs += [('out', E.undo(a, f)), ('own0', E.own(a, f, 0.0)), ('own05', E.own(a, f, 0.5)), ('ownfine', E.own(a, f, 1.0, 0.5, 1.0))]
        for tag, ref in refs:
            d = db(np.fromfile(pre + '.%s.f64' % tag).reshape(2, len(ref)).T, ref); worst = max(worst, d); out.append('%s %.0f' % (tag, d))
        print('     app against the prototype with the same settings, dB: ' + ', '.join(out), flush=True)
    for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp); print('%d pairs (%d with a crusher), %d with different answers; the sounds differ by at most %.0f dB' % (len(paths) // 2, found, bad, worst)); sys.exit(1 if bad or worst > -120.0 else 0)
main()
