# Checks of the impulse-response export and of the fader positions (one process, about a minute).
import sys, os, json
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/tmp/claude-0/v2/tone')
import numpy as np, scipy.signal as ss
from base import *
import tone
FB = tone.FB; NF = 1 << 18; fd = np.fft.rfftfreq(NF, 1 / SR); band = (fd >= 20) & (fd <= 20000)
def curves():
    lf = np.log2(np.maximum(FB, 1.0) / 1000.0); c = {}
    c['smooth EQ (+9 dB bell at 300 Hz, -12 dB bell at 3 kHz, +6 dB shelf)'] = 9 * np.exp(-0.5 * ((lf - np.log2(0.3)) / 0.4) ** 2) - 12 * np.exp(-0.5 * ((lf - np.log2(3.0)) / 0.25) ** 2) + 6 / (1 + np.exp(-(lf - 3) * 4))
    c['reflection, gain 0.7 at 97.3 samples'] = tone.comb_db(97.3, 0.7); c['reflection, gain -0.9 at 300.5 samples'] = tone.comb_db(300.5, -0.9)
    d = [x for x in case_dirs() if 'tone_box' in x][0]; f, db = true_db(load(d, 'A')); c['a true box curve (dev case), on the bins'] = cell_mean(f, db[0], FB)
    return c
if __name__ == '__main__':
    print('1. Impulse response against the curve it was made from: error of its magnitude response, dB, 20 Hz .. 20 kHz: rms / largest')
    for name, c in curves().items():
        want = tone._to_bins(FB, c, fd); print('  ' + name)
        for ph in ('minimum', 'linear'):
            s = '    %-8s' % ph
            for n in (2048, 4096, 8192, 16384, 32768):
                h = tone.to_ir(FB, c, n, ph); e = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(h, NF)), 1e-9)) - want; s += ' | %5d taps %.2f / %.1f' % (n, np.sqrt((e[band] ** 2).mean()), np.abs(e[band]).max())
            print(s)
    c = curves()['reflection, gain 0.7 at 97.3 samples']; h = tone.to_ir(FB, c, 16384, 'linear'); m = 8192; print('2. linear phase: largest difference between the two halves around tap n/2: %.1e (largest tap %.2f at %d)' % (np.abs(h[m + 1:m + 8000] - h[m - 1:m - 8000:-1]).max(), np.abs(h).max(), int(np.argmax(np.abs(h)))))
    h = tone.to_ir(FB, c, 16384, 'minimum'); e = np.cumsum(h ** 2) / (h ** 2).sum(); print('   minimum phase: largest tap %.2f at %d; share of the energy in the first 128 taps %.4f; two rows in -> shape %s' % (np.abs(h).max(), int(np.argmax(np.abs(h))), e[127], tone.to_ir(FB, np.stack([c, c]), 4096).shape))
    g24 = tone.GRID; h = tone.to_ir(g24, np.interp(np.log(g24), np.log(np.maximum(FB, 1)), curves()['smooth EQ (+9 dB bell at 300 Hz, -12 dB bell at 3 kHz, +6 dB shelf)']), 8192); print('   a curve on another grid (240 points, 1/24 octave) is accepted: %d taps' % len(h))
    print('3. Fader positions against the truth (the effect really applied a second time, with its true impulse response)')
    for key in ('tone_comb', 'tone_pb_eq', 'tone_calf_eq'):
        for d in [x for x in case_dirs() if key in x][:2]:
            a = load(d, 'A'); b = load(d, 'B'); f, db = true_db(a); tc = cell_mean(f, db.mean(0), FB); g, bc = tone.contour(a['wet'])
            twice = lambda x: np.stack([ss.fftconvolve(x[:, o], a['h'][o][:, o])[:len(x)] for o in range(2)], 1)
            aw2 = twice(a['wet']); bw2 = twice(b['wet']); s2 = score.Scorer(aw2); sb2 = score.Scorer(bw2); sd = score.Scorer(a['dry']); sb = score.Scorer(b['wet']); row = '  %s:' % a['name']
            for lab, cv in (('true curve', tc), ('blind curve', bc)):
                row += ' | %s: fader 0 removed %.0f%%, fader 2 doubled %.0f%%, second sound at 1: %.0f%%, at 2: %.0f%%' % (lab, 100 * (1 - sd(tone.apply_curve(a['wet'], FB, cv, 0.0)) / sd(a['wet'])), 100 * (1 - s2(tone.apply_curve(a['wet'], FB, cv, 2.0)) / s2(a['wet'])),
                        100 * (1 - sb(tone.put_curve(b['dry'], FB, cv, 1.0)) / sb(b['dry'])), 100 * (1 - sb2(tone.put_curve(b['dry'], FB, cv, 2.0)) / sb2(b['dry'])))
            print(row)
            if key == 'tone_comb': print('      fader on the wet sound, distance to the dry one at 0, 0.25, 0.5, 0.75, 1:', ' '.join('%.3f' % sd(tone.apply_curve(a['wet'], FB, bc, am)) for am in (0, 0.25, 0.5, 0.75, 1.0)), '| fader 1 returns the sound itself:', bool(np.array_equal(tone.apply_curve(a['wet'], FB, bc, 1.0), np.asarray(a['wet'], np.float64))))
