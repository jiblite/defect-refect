# The echo faders on the sound itself: what does the engine find on the result?
import sys, json, subprocess, numpy as np, soundfile as sf
tool = '/tmp/claude-0/drbuild/drecho'; D = '/dev/shm/ep/'
for name in sys.argv[1:]:
    src = D + name + '_wet.wav'; x, _ = sf.read(src, dtype='float64'); dry, _ = sf.read(D + name + '_dry.wav', dtype='float64')
    def lv(a, b): return 10 * np.log10(max(((a - b) ** 2).sum(), 1e-30) / max((b ** 2).sum(), 1e-30))
    for fad in (['amount=1'], ['amount=0'], ['amount=0.5'], ['amount=2'], ['amount=1', 'time=1.5'], ['amount=1', 'feedback=0'], ['amount=1', 'feedback=2'], ['amount=1', 'bright=0'], ['amount=1.5', 'time=0.7', 'feedback=1.5', 'bright=0.5']):
        p = subprocess.run([tool, 'out', src, D + 'own.wav'] + fad, capture_output=True, text=True); y, _ = sf.read(D + 'own.wav', dtype='float64')
        M = json.loads(subprocess.run([tool, 'measure', D + 'own.wav'], capture_output=True, text=True).stdout); f = M['ch'][0]['found']
        print('%-22s %-48s against the input %6.1f dB, against the dry sound %6.1f dB, peak %.2f | found on the result: %s   (%s)' % (name, ' '.join(fad), lv(y, x), lv(y, dry), np.abs(y).max(),
              ('%.1f ms p %.2f levels %s' % (f['q'] / 48.0, f['p'], ' '.join('%.1f' % v for v in f['level'][:6]))) if f else 'no echo (sure %.2f)' % M['sure'], p.stderr.strip().split(', ')[-1]))
