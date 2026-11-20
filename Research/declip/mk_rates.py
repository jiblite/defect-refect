# Test files for the declipper at other rates and with one channel: a clean sound brought to the rate, then cut flat
# there (a plain cut that takes about 5 % of the values; a soft one: tanh at 20 dB of drive). Container only.
#   python3 mk_rates.py out_dir name [name ...]      (clean sounds: /dev/shm/dall/name_dry.wav, 48 kHz)
import sys, os, numpy as np, soundfile as sf, scipy.signal as ss
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for name in sys.argv[2:]:
    dry = sf.read('/dev/shm/dall/' + name + '_dry.wav', dtype='float64', always_2d=True)[0]
    for tag, rate, ch in (('44', 44100, 2), ('96', 96000, 2), ('mono', 48000, 1), ('48', 48000, 2)):
        d = dry if rate == 48000 else ss.resample_poly(dry, rate // 300, 160, axis=0); d = d / np.abs(d).max()
        if ch == 1: d = d[:, :1]
        c = np.quantile(np.abs(d), 0.95); hard = np.clip(d, -c, c); soft = np.tanh(10.0 * d)
        for kind, w in (('hard', hard), ('soft', soft)):
            pre = os.path.join(out, '%s_%s_%s' % (name, kind, tag)); sf.write(pre + '_wet.wav', 0.98 * w / np.abs(w).max(), rate, subtype='PCM_24')
            ref = dry / np.abs(dry).max(); ref = np.repeat(ref[:, :1], 2, 1) if ch == 1 else ref; sf.write(pre + '_dry.wav', ref, 48000, subtype='PCM_24')
        print(name, tag, d.shape, 'cut at %.3f of the peak' % c, flush=True)
