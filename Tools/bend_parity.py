# The bend engine in C++ (Source/Core/Bend.h, through drbend) against the Python prototype it was ported from
# (Research/bend/bfeat.py, bengine.py, which read the same model file with plain numpy): the same numbers, the same sounds?
#   python3 Tools/bend_parity.py path/to/drbend Resources/bend.bin a.wav b.wav [a2.wav b2.wav ...]
# Every pair: the bend is read off a, taken out of it (all the way, half way) and pushed further, and put onto b.
# 48 kHz WAVs. Prints the numbers side by side; exit code 1 if what the meter sees differs by more than 1e-4, its
# answer by more than 0.01 dB, or a sound is further than -100 dB from the prototype's.
import sys, os, subprocess, tempfile, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'bend')]
import bfeat, bengine as E

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
        a = rd(pa); b = rd(pb); X, vp = bfeat.features(a); F = E.find(a, model); out, y = app(pa, '-', 0.0, len(a)); Q = np.fromfile(pre + '.feat.f64'); B = np.fromfile(pre + '.bend.f64'); n = 2 * bfeat.NF; g = 2 * len(bfeat.GRID)
        df = float(np.abs(Q[:n] - X.ravel()).max()); dp = float(np.abs(Q[n:n + 2] / vp - 1).max()); dS = float(np.abs(B[:g] - F['S'].ravel()).max()); dl = abs(B[g] - F['logit']); dt = abs(B[g + 1] - F['top'])
        print('%s: %s' % (os.path.basename(pa), out.splitlines()[0])); print('  what the meter sees differs by %.2e at the most, the peaks by %.2e; its answer by %.2e dB, the logit by %.2e (prototype %.3f), the squeeze at the peak by %.2e' % (df, dp, dS, dl, F['logit'], dt))
        if df > 1e-4 or dp > 1e-9 or dS > 0.01 or dl > 0.01: bad += 1; print('  DIFFERENT ANSWERS')
        for tag, s in (('out', 0.0), ('half', 0.5), ('further', 1.5), ('twice', 2.0)):
            d = db(app(pa, '-', s, len(a))[1], E.own(a, F['S'], F['vp'], s)); worst = max(worst, d); print('  %-10s app against prototype: %.1f dB' % (tag, d), flush=True)
        for tag, s in (('onto', 1.0), ('onto half', 0.5), ('onto twice', 2.0)):
            d = db(app(pa, pb, s, len(b))[1], E.onto(b, F['S'], F['vp'], s)); worst = max(worst, d); print('  %-10s app against prototype: %.1f dB' % (tag, d), flush=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    print('%d pairs, %d with different answers; the sounds differ by at most %.0f dB' % (len(paths) // 2, bad, worst)); sys.exit(1 if bad or worst > -100.0 else 0)
main()
