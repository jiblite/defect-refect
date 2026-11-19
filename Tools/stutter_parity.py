# The stutter engine in C++ (Source/Core/Stutter.h, through drstutter) against the Python prototype it was ported
# from (Research/stutter/stutter.py): the same events, the same sounds?
#   python3 Tools/stutter_parity.py path/to/drstutter a.wav b.wav [a2.wav b2.wav ...]
# Every pair: the events are read off a, played on b, and a itself is rendered with the faders moved. Prints the
# numbers side by side; exit code 1 if anything read differs or a sound is further than -200 dB from the prototype's.
import sys, os, json, subprocess, tempfile, time, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'stutter')]
import stutter as E

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
        ev = [list(e) for e in F['events']]; lo = [list(e) for e in F['left_out']]; st = [list(e) for e in F.get('stretches', [])]
        same = bool(F['found']) == J['found'] and ev == J['events'] and lo == J['left_out'] and list(F['own']) == J['own'] and st == J['stretches'] and (not F['found'] or (bool(F['fades']) == J['fades'] and abs(F['share'] - J['share']) < 1e-9))
        print('%s: app %.2f s (find %.3f s), prototype %.2f s | found %s / %s, events %d / %d, left out %d / %d, stretches %d / %d, fades %s / %s%s' % (os.path.basename(pa), t1 - t0, J['took'], t3 - t2, F['found'], J['found'], len(ev), len(J['events']), len(lo), len(J['left_out']), len(st), len(J['stretches']), F.get('fades'), J['fades'], '' if same else '   DIFFERENT ANSWERS'), flush=True)
        bad += int(not same)
        if not (F['found'] and J['found'] and same): continue
        found += 1; n = len(a); d1 = float(np.abs(np.fromfile(pre + '.same.f64').reshape(2, n).T - a).max()); bad += int(d1 != 0.0); out = ['middle %.0e' % d1]
        for tag, ref in (('onto', E.onto(b, F, 1.0)), ('onto2', E.onto(b, F, 2.0)), ('onto05', E.onto(b, F, 0.5)), ('fine', E.onto(b, F, 1.0, 0.5)), ('own2', E.own(a, F, 2.0)), ('own0', E.own(a, F, 0.0)), ('own05', E.own(a, F, 0.5)), ('ownfine', E.own(a, F, 1.0, 0.5))):
            d = db(np.fromfile(pre + '.%s.f64' % tag).reshape(2, len(ref)).T, ref); worst = max(worst, d); out.append('%s %.0f' % (tag, d))
        print('     app against prototype, dB: ' + ', '.join(out), flush=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp); print('%d pairs (%d with a stutter), %d with different answers; the sounds differ by at most %.0f dB' % (len(paths) // 2, found, bad, worst)); sys.exit(1 if bad or worst > -200.0 else 0)
main()
