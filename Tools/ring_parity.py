# The ring engine in C++ (Source/Core/Ring.h, through drring) against the Python prototype it was ported from
# (Research/ring/ring.py): the same verdict, the same tone, the same sounds?
#   python3 Tools/ring_parity.py path/to/drring a.wav b.wav [a2.wav b2.wav ...]
# Every pair: the tone is read off a, taken out of it, and the gain is put onto b. Prints the numbers side by side;
# exit code 1 if a verdict or the tone (pitch, phase, how much plain sound beside it) differs or a sound is further
# than -100 dB from the prototype's.
import sys, os, json, subprocess, tempfile, time, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'ring')]
import ring as E

def db(a, b): return 10 * np.log10(((a - b) ** 2).mean() / max((b ** 2).mean(), 1e-30) + 1e-30)
def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)

def main():
    exe = sys.argv[1]; paths = sys.argv[2:]; tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); pre = os.path.join(tmp, 'x'); bad = 0; worst = -999.0
    for pa, pb in zip(paths[0::2], paths[1::2]):
        a = rd(pa); b = rd(pb); t0 = time.time(); r = subprocess.run([exe, pa, pb, pre], capture_output=True, text=True); t1 = time.time()
        if r.returncode: print(r.stderr); bad += 1; continue
        J = json.loads(r.stdout.strip().splitlines()[-1]); t2 = time.time(); F = E.find(a); t3 = time.time()
        print('%s: app %.2f s (find %.2f s), prototype %.1f s' % (os.path.basename(pa), t1 - t0, J['took'], t3 - t2), flush=True)
        same = bool(F['found']) == J['found'] and ('first_win' in F) == J['looked'] and ('f' in F) == J['measured']
        print('  prototype: found %s | app: found %s' % (F['found'], J['found']))
        if 'first_win' in F and J['looked']:
            print('  first look: win %.6f / %.6f at %.4f / %.4f Hz, rho %.4f / %.4f' % (F['first_win'], J['first_win'], F['first_f'], J['first_f'], F['first_rho'], J['first_rho'])); same = same and abs(F['first_win'] - J['first_win']) < 1e-6 and abs(F['first_f'] - J['first_f']) < 1e-6
        if 'f' in F and J['measured']:
            dph = float(np.angle(np.exp(1j * (F['phase'] - J['phase']))))
            print('  the tone: %.7f / %.7f Hz, phase apart %.1e, rho %.7f / %.7f, floor %.3e / %.3e, passes %d / %d' % (F['f'], J['f'], dph, F['rho'], J['rho'], F['eps'], J['eps'], F['passes'], J['passes']))
            print('     win %.5f / %.5f  match %.4f / %.4f dB  dip %.3f / %.3f dB  offset %.2f / %.2f dB  holes %.5f / %.5f' % (F['win'], J['win'], F['match'], J['match'], F['dip'], J['dip'], F['offset'], J['offset'], F['holes'], J['holes']))
            same = same and abs(F['f'] - J['f']) < 1e-6 and abs(dph) < 1e-6 and abs(F['rho'] - J['rho']) < 1e-6 and abs(np.log(F['eps'] / J['eps'])) < 1e-3 and abs(F['holes'] - J['holes']) < 1e-9
        bad += int(not same)
        if not same: print('  DIFFERENT ANSWERS')
        if not (F['found'] and J['found'] and same): continue
        n = len(a); d = db(np.fromfile(pre + '.x.f64').reshape(2, n).T, F['x']); worst = max(worst, d); print('  %-6s app against prototype: %.1f dB' % ('x', d))
        d1 = float(np.abs(np.fromfile(pre + '.same.f64').reshape(2, n).T - a).max()); print('  Amount in the middle: the largest difference from the sound itself %.1e' % d1); bad += int(d1 != 0.0)
        for tag, ref in (('out', E.own(a, F, 0.0)), ('half', E.own(a, F, 0.5)), ('twice', E.own(a, F, 2.0)), ('up', E.own(a, F, 1.0, 1.5)), ('onto', E.onto(b, F, 1.0)), ('onto2', E.onto(b, F, 2.0)), ('onto05', E.onto(b, F, 0.5))):
            d = db(np.fromfile(pre + '.%s.f64' % tag).reshape(2, len(ref)).T, ref); worst = max(worst, d); print('  %-6s app against prototype: %.1f dB' % (tag, d), flush=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp); print('%d pairs, %d with different answers; the sounds differ by at most %.0f dB' % (len(paths) // 2, bad, worst)); sys.exit(1 if bad or worst > -100.0 else 0)
main()
