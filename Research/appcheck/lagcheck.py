# lagcheck.py export.wav ref.wav: how many samples the app's export comes after the render test's output, and how equal they are once lined up
import sys, numpy as np, soundfile as sf
from scipy import signal
a, sra = sf.read(sys.argv[1], dtype='float64'); b, srb = sf.read(sys.argv[2], dtype='float64'); assert sra == srb == 48000
n = min(len(a), len(b)); x, y = a[:n, 0], b[:n, 0]
c = signal.correlate(x, y, mode='full', method='fft'); lag = int(np.argmax(np.abs(c))) - (n - 1)
def diff(lag):
    if lag >= 0: u, v = a[lag:n], b[:n - lag]
    else: u, v = a[:n + lag], b[-lag:n]
    g = float(np.sum(u * v) / max(np.sum(v * v), 1e-30)); return g, 10 * np.log10(np.sum((u - g * v) ** 2) / max(np.sum((g * v) ** 2), 1e-30) + 1e-30)
print('%s: %d samples at 48 kHz, export against the render test: lag %d samples (%.2f ms)' % (sys.argv[1].split('/')[-1], len(a), lag, 1000.0 * lag / 48000))
print('   unshifted: gain %.4f, difference %.1f dB | lined up: gain %.4f, difference %.1f dB' % (diff(0) + diff(lag)))
