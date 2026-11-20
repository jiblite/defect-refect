# The filter engine in C++ (Source/Core/Filter.h, through drfilter) against the Python prototype it was ported from
# (Research/filter/fengine.py, which reads the same model file with plain numpy): the same curve, the same sounds?
#   python3 Tools/filter_parity.py path/to/drfilter Resources/filter.bin a.wav b.wav [a2.wav b2.wav ...]
# Every pair: the filter is read off a, taken out of it (all the way, half way, doubled, and with the Cutoff and
# Resonance faders moved), and put onto b. 48 kHz WAVs. Prints the numbers side by side; exit code 1 if a curve is
# further than 0.01 dB from the prototype's, a lift cap differs, or a sound is further than -100 dB from the prototype's.
import sys, os, subprocess, tempfile, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'filter')]
import fengine as E

def db(a, b): return 10 * np.log10(((a - b) ** 2).mean() / max((b ** 2).mean(), 1e-30) + 1e-30)
def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)

def main():
    exe, model = sys.argv[1], sys.argv[2]; paths = sys.argv[3:]; tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); pre = os.path.join(tmp, 'x'); bad = 0; worst = -999.0; env = dict(os.environ, DR_FILTER_ANY='1')
    def app(pa, pb, amount, cutoff, res, n):
        r = subprocess.run([exe, model, pa, pb, pre, str(amount), str(cutoff), str(res)], capture_output=True, text=True, env=env)
        if r.returncode: raise RuntimeError(r.stderr)
        return r.stdout, np.fromfile(pre + '.out.f64').reshape(n, 2)
    for pa, pb in zip(paths[0::2], paths[1::2]):
        a = rd(pa); b = rd(pb); F = E.find(a, model); out, y = app(pa, '-', 0.0, 1.0, 1.0, len(a)); Q = np.fromfile(pre + '.curve.f64'); n = E.ffeat.NBAND
        dc = float(np.abs(Q[:n] - F['curve']).max()); dl = float(np.abs(Q[n:2 * n] - F['lift']).max()); da = float(np.abs(Q[2 * n:3 * n] - F['apart']).max()); ds = abs(Q[3 * n] - F['sure'])
        print('%s: %s' % (os.path.basename(pa), out.splitlines()[0])); print('  curve differs by %.2e dB at the most, lift caps by %.2e, nets apart by %.2e, the logit by %.2e (prototype %.3f)' % (dc, dl, da, ds, F['sure']))
        if dc > 0.01 or dl > 1e-9 or ds > 0.01: bad += 1; print('  DIFFERENT ANSWERS')
        oct_of = lambda fader: 2.0 * (min(max(fader, 0.0), 2.0) - 1.0)
        for tag, (s, cut, res) in (('out', (0.0, 1.0, 1.0)), ('half', (0.5, 1.0, 1.0)), ('twice', (2.0, 1.0, 1.0)), ('moved', (1.0, 1.25, 0.5)), ('moved, out half', (0.5, 0.8, 2.0))):
            d = db(app(pa, '-', s, cut, res, len(a))[1], E.own(a, F, s, oct_of(cut), res)); worst = max(worst, d); print('  %-16s app against prototype: %.1f dB' % (tag, d), flush=True)
        for tag, (s, cut, res) in (('onto', (1.0, 1.0, 1.0)), ('onto twice', (2.0, 1.0, 1.0)), ('onto, moved', (1.0, 0.7, 1.5))):
            d = db(app(pa, pb, s, cut, res, len(b))[1], E.onto(b, F, s, oct_of(cut), res)); worst = max(worst, d); print('  %-16s app against prototype: %.1f dB' % (tag, d), flush=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    print('%d pairs, %d with different answers; the sounds differ by at most %.0f dB' % (len(paths) // 2, bad, worst)); sys.exit(1 if bad or worst > -100.0 else 0)
main()
