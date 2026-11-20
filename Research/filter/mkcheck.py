# Sounds for the memory check's stage of the filter engine and the bend engine (Research/moving/memcheck.py four):
# long ones, ones at the limits of what is read, files at other sample rates, mono files, and one sound with both.
#   python3 mkcheck.py            writes into DR_FOUR_MC (default /tmp/claude-0/v2/four/mc/)
import sys, os, numpy as np, soundfile as sf, scipy.signal as ss
sys.path.insert(0, os.environ.get('DR_PROTO', '/tmp/claude-0/v2/proto')); import truth
SETS = os.environ.get('DR_SETS', '/tmp/claude-0/v2/bench/sets/'); D = os.environ.get('DR_FOUR_MC', '/tmp/claude-0/v2/four/mc/'); os.makedirs(D, exist_ok=True)
def w(name, x, sr=48000, sub='FLOAT'): sf.write(D + name + '.wav', x, sr, subtype=sub); print(name, len(x), sr)
def at(x, sr): g = np.gcd(sr, 48000); return ss.resample_poly(x, sr // g, 48000 // g, axis=0)
fil = truth.load(SETS + 'filter/fil_calf_00/', 'A'); dst = truth.load(SETS + 'distortion/dist_sox_00/', 'A'); sat = truth.load(SETS + 'saturation/sat_calf_tape_00/', 'A')
bend = lambda x, drive: np.tanh(x / max(np.abs(x).max(), 1e-9) * 10 ** (drive / 20)) * 0.5
w('filtLong', np.tile(fil['wet'], (7, 1)))                                  # 42 s: read in every fourth frame
w('filt1s', fil['wet'][48000:96000]); w('filt0s9', fil['wet'][48000:48000 + 45000])      # 20 frames: just read; 18 frames: not read
w('filt44', at(fil['wet'], 44100), 44100); w('filtMono', fil['wet'].mean(1))
w('bendLong', np.tile(dst['wet'], (7, 1))); w('bendShort', dst['wet'][48000:48000 + 20000]); w('bendHalf', dst['wet'][48000:48000 + 24000])      # under half a second: not read; just half a second: read
for sr in (44100, 96000, 192000): w('bend%d' % (sr // 1000), bend(at(dst['dry'], sr), 18.0), sr)            # bent at the file's own rate: read there (192 kHz: every fourth sample)
w('bend44pcm', bend(at(sat['dry'], 44100), 8.0), 44100, 'PCM_16'); w('bendMono', bend(dst['dry'].mean(1), 14.0))
lp = ss.butter(4, 1800 / 24000, output='sos'); w('both', bend(ss.sosfilt(lp, fil['dry'], axis=0), 12.0))     # a low-pass, then a bend: both groups on one sound
w('bothOther', ss.sosfilt(lp, bend(fil['dry'], 12.0) , axis=0))                                             # a bend, then a low-pass: the bend hidden behind the filter
