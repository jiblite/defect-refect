# The declipper in C++ (Source/Core/Declip.h and its place in Bend.h, through drbend) against the Python prototype it was
# ported from (Research/declip/dengine.py): the same verdicts, the same numbers, the same sounds?
#   python3 Tools/declip_parity.py path/to/drbend Resources/bend.bin a.wav b.wav [a2.wav b2.wav ...]
# Every pair: a's flat tops are looked for and rebuilt, the fader is set far left, half way, further and far right, and
# the found bend is put onto b. 48 kHz WAVs (16 or 24 bit, or 32 bit float). Prints the numbers side by side; exit code 1
# if a verdict differs (what kind of top each half-wave has, whether it was rebuilt, whether the second pass was kept,
# whether the bend shows), the self-test's score by more than 1, a curve's top by more than 0.1 dB, or a sound is
# further than -60 dB from the prototype's (the rebuilding is hundreds of rounds long: the two do not agree to the last
# digit, and need not).
import sys, os, subprocess, tempfile, time, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'bend'), os.path.join(HERE, '..', 'Research', 'declip')]
import bfeat, bengine as E, dengine as D

def db(a, b): return 10 * np.log10(((a - b) ** 2).mean() / max((b ** 2).mean(), 1e-30) + 1e-30)
def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)

def main():
    exe, model = sys.argv[1], sys.argv[2]; paths = sys.argv[3:]; tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); pre = os.path.join(tmp, 'x'); bad = 0; worst = -999.0; env = dict(os.environ, DR_BEND_ANY='1')
    def app(pa, pb, amount, n):
        r = subprocess.run([exe, model, pa, pb, pre, str(amount)], capture_output=True, text=True, env=env)
        if r.returncode: raise RuntimeError(r.stderr)
        return r.stdout, np.fromfile(pre + '.out.f64').reshape(n, 2)
    for pa, pb in zip(paths[0::2], paths[1::2]):
        a = rd(pa); b = rd(pb); F = E.find(a, model); t0 = time.time(); A = D.analyse(a, F['S'], F['vp'], F['found']); V = D.verdict(F['logit'], F['top'], A); sec = time.time() - t0
        r = subprocess.run([exe, model, pa, '-', pre, '0'], capture_output=True, text=True, env=dict(os.environ)); lines = r.stdout.splitlines(); w = lines[0].split(); fl = [l for l in lines if l.startswith('flat:')][0].split()
        g = lambda key, src=fl: src[src.index(key) + 1]; kinds = '/'.join(A['kinds']); tops = [(-1.0 if (not A['did'] or A['tabs'][h] is None) else A['tabs'][h].top) for h in (0, 1)]; at = [float(v) for v in fl[fl.index('tops') + 1:fl.index('tops') + 3]]
        print('%s: prototype %s did %d rebuilt %d second %d test %s ratio %s cut %.6f tops %.3f %.3f | %s sure %.3f %s top %.2f (%.0f s)' % (os.path.basename(pa), kinds, A['did'], A['rebuilt'], A['second'], 'none' if A['test'] is None else '%.2f' % A['test'],
              'none' if A['ratio'] is None else '%.4f' % A['ratio'], A['cut'], tops[0], tops[1], 'found' if V['found'] else 'not found', V['sure'], V['kind'], V['top'], sec))
        print('  app: %s\n       %s\n       %s' % (lines[0], ' '.join(fl), lines[-1]))
        same = (g('kinds') == kinds and int(g('did')) == int(A['did']) and int(g('rebuilt')) == int(A['rebuilt']) and int(g('second')) == int(A['second']) and int(w[1]) == int(V['found']) and w[10] == V['kind'] and abs(float(w[5]) - V['sure']) < 2e-3 and abs(float(w[7]) - V['top']) < 0.11
                and abs(float(g('cut')) - A['cut']) < 2e-6 and (A['test'] is None or abs(float(g('test')) - A['test']) < 1.0) and (A['ratio'] is None or abs(float(g('ratio')) - A['ratio']) < 0.02) and max(abs(at[0] - tops[0]), abs(at[1] - tops[1])) < 0.1)
        if not same: bad += 1; print('  DIFFERENT ANSWERS')
        for tag, s in (('out', 0.0), ('half', 0.5), ('further', 1.5), ('twice', 2.0)):
            d = db(app(pa, '-', s, len(a))[1], D.own(a, F['S'], F['vp'], A, s)); worst = max(worst, d); print('  %-10s app against prototype: %.1f dB' % (tag, d), flush=True)
        for tag, s in (('onto', 1.0), ('onto half', 0.5), ('onto twice', 2.0)):
            d = db(app(pa, pb, s, len(b))[1], D.onto(b, F['S'], F['vp'], A, s)); worst = max(worst, d); print('  %-10s app against prototype: %.1f dB' % (tag, d), flush=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    print('%d pairs, %d with different answers; the sounds differ by at most %.0f dB' % (len(paths) // 2, bad, worst)); sys.exit(1 if bad or worst > -60.0 else 0)
main()
