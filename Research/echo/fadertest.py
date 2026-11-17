# The echo engine's faders, checked on the filter it makes (drecho ir): where the repeats sit and how loud they are.
import sys, os, json, subprocess, numpy as np, soundfile as sf
tool = '/tmp/claude-0/drbuild/drecho'; D = '/dev/shm/ep/'
def ir(src, *fad):
    subprocess.run([tool, 'ir', src, D + 'ir'] + list(fad), capture_output=True, text=True, check=True)
    return [np.fromfile(D + 'ir.%s.f64' % c) for c in 'LR']
def reps(h, q, kmax=10):
    out = []
    for k in range(1, kmax + 1):
        a = k * q - q // 2; b = k * q + (q + 1) // 2
        if a >= len(h): break
        e = (h[a:b] ** 2).sum(); out.append(10 * np.log10(max(e, 1e-30)))
        if out[-1] < -70: out.pop(); break
    return out
def cent(h, q):
    a = q - q // 2; b = min(len(h), q + (q + 1) // 2); seg = h[a:b]; H = np.abs(np.fft.rfft(seg, 8192)) ** 2; f = np.fft.rfftfreq(8192, 1 / 48000.0)
    lo = H[(f > 100) & (f < 1000)].mean(); hi = H[(f > 4000) & (f < 12000)].mean(); return 10 * np.log10(hi / lo)
for name in sys.argv[1:]:
    src = D + name + '_wet.wav'; M = json.loads(subprocess.run([tool, 'measure', src], capture_output=True, text=True).stdout); f = M['ch'][0]['found']
    if not f: print(name, 'nothing found'); continue
    q = f['q']; print('%s: left echo %.1f ms, phi %.3f, levels as found %s' % (name, q / 48.0, f['phi'], ' '.join('%.1f' % v for v in f['level'][:8])))
    for fad in ([], ['amount=2'], ['amount=0.5'], ['time=1.5'], ['time=0.5'], ['feedback=0'], ['feedback=0.5'], ['feedback=1.5'], ['feedback=2'], ['bright=0'], ['bright=0.5'], ['bright=2'], ['time=2', 'feedback=2', 'bright=0.3', 'amount=1.5']):
        h = ir(src, *fad)[0]; t = 1.0
        for a in fad:
            if a.startswith('time='): t = float(a[5:])
        qq = int(round(q * t)); r = reps(h, qq)
        pk = int(np.argmax(np.abs(h[qq - 200:qq + 200]))) + qq - 200
        print('   %-48s len %7d  h[0] %.4f  first repeat at %.1f ms  levels %s  | tone of 1st (hi-lo dB) %+.1f' % (' '.join(fad) or 'as found', len(h), h[0], pk / 48.0, ' '.join('%.1f' % v for v in r[:9]), cent(h, qq)))
