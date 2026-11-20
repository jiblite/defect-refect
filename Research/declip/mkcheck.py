# Sounds for the memory check's stage of the declipper (Research/moving/memcheck.py flat): plain cuts and soft flat
# tops that are rebuilt, ones that each safeguard refuses, long ones, ones at the limits of what is read, files at other
# sample rates, mono files, and sounds an engine could trip over (nothing known, one side or one channel cut, a cut that
# holds for a second, two levels of cut).
#   python3 mkcheck.py            writes into DR_DECLIP_MC (default /tmp/claude-0/v2/declip/mc/)
# Five cases of the 162-case trial that take a rare path are copied in as they are when their folder is there
# (DR_DECLIP_TRIAL, default /dev/shm/); the folder holds audio and stays on the machine.
import sys, os, shutil, numpy as np, soundfile as sf, scipy.signal as ss
sys.path.insert(0, os.environ.get('DR_PROTO', '/tmp/claude-0/v2/proto')); import truth
SETS = os.environ.get('DR_SETS', '/tmp/claude-0/v2/bench/sets/'); D = os.environ.get('DR_DECLIP_MC', '/tmp/claude-0/v2/declip/mc/'); T = os.environ.get('DR_DECLIP_TRIAL', '/dev/shm/'); os.makedirs(D, exist_ok=True)
def w(name, x, sr=48000, sub='PCM_24'): sf.write(D + name + '.wav', x, sr, subtype=sub); print(name, len(x), sr, flush=True)
def at(x, sr): g = np.gcd(sr, 48000); return ss.resample_poly(x, sr // g, 48000 // g, axis=0)
def unit(x): return x / max(np.abs(x).max(), 1e-9)
def cut(x, share=0.05, lv=0.7):         # a plain cut that takes this share of the values, the cut at lv of full scale
    x = unit(x); c = np.quantile(np.abs(x), 1.0 - share); return np.clip(x, -c, c) * (lv / c)
def soft(x, drive=20.0): return 0.9 * np.tanh(unit(x) * 10 ** (drive / 20))
def over(x, drive, k=4): return 0.9 * unit(ss.resample_poly(np.tanh(ss.resample_poly(unit(x), k, 1, axis=0) * 10 ** (drive / 20)), 1, k, axis=0))     # a clipper that works at k times the rate: ripple on the tops
P = truth.load(SETS + 'distortion/dist_pedalboard_00/', 'A')['dry']; Q = truth.load(SETS + 'distortion/dist_sox_00/', 'A')['dry']; rng = np.random.default_rng(52); n = len(P)
# rebuilt (cut, soft); a quarter of the values cut, which the self-test refuses (cutDeep); a light soft clipping without a pile at the top, nothing to rebuild (softOther)
w('cut', cut(P)); w('cutDeep', cut(Q, 0.25)); w('soft', soft(P)); w('softOther', soft(Q, 14.0))
# refused: a dense sound (the self-test), nearly a square wave, a clipper at four times the rate (its second pass)
w('cutNoise', cut(rng.standard_normal((n, 2)), 0.3)); w('softSquare', soft(P, 50.0)); w('softOver', over(P, 20.0)); w('softOver2', over(Q, 26.0, 2))
# long: many stretches on all threads; a soft top's first pass on parts of the sound; the same where the second pass may go
w('cutLong', cut(np.tile(P, (16, 1)))); w('softLong', soft(np.vstack([P, Q] * 5))); w('softOverLong', over(np.vstack([P, Q] * 4), 20.0))
# limits of what is read: half a second and just under it, a second
w('cutHalf', cut(P[48000:48000 + 24000])); w('cutShort', cut(P[48000:48000 + 23999])); w('cut1s', cut(P[48000:96000])); w('soft1s', soft(P[48000:96000]))
# other rates (cut at the file's own rate), mono, 16 bit
for sr in (22050, 44100, 96000, 192000): w('cut%d' % (sr // 1000), cut(at(P, sr)), sr)
w('soft44', soft(at(P, 44100)), 44100); w('soft96', soft(at(Q, 96000)), 96000); w('cutMono', cut(P.mean(1))); w('softMono', soft(Q.mean(1))); w('cut44pcm', cut(at(Q, 44100)), 44100, 'PCM_16')
# sounds to trip over
t = np.arange(n) / 48000.0; sq = 0.5 * np.sign(np.sin(2 * np.pi * 110.0 * t)); w('square', np.stack([sq, sq], 1))                       # nothing known: every value at the top
x = unit(P); c = np.quantile(np.abs(x), 0.95); w('cutUp', np.minimum(x, c)); w('cutLeft', np.stack([np.clip(x[:, 0], -c, c), np.zeros(n)], 1))    # one side only; one channel only, the other silent
w('cutTwo', np.stack([np.clip(x[:, 0], -c, c), np.clip(x[:, 1], -0.6 * c, 0.6 * c)], 1))                                            # another level of cut on each channel
y = cut(P); y[96000:144000] = np.abs(y).max(); w('cutHeld', y)                                                                      # the cut holds for a second
y = cut(P); w('cutGap', np.vstack([y, np.zeros((480000, 2)), y]))                                                                   # ten seconds of nothing between two takes
y = cut(P); y[:, 1] = y[:, 0]; w('cutTwin', y); y = y.copy(); y[::1000, 1] *= 0.999; w('cutNearTwin', y)                            # both channels alike, and all but
w('cutFull', np.clip(unit(P) * 4.0, -1.0, 1.0), sub='PCM_16')                                                                       # into full scale of a 16 bit file
z = unit(np.random.default_rng(53).standard_normal((n, 2))); c = np.quantile(z, 0.75); z = np.minimum(z, c); w('cutNoiseUp', 0.9 * z / np.abs(z).max())      # a dense sound cut on one side only: refused, the cut half stays as it is and the other is walked back the meter's way
w('softMonoP', soft(P.mean(1))); w('soft96P', soft(at(P, 96000)), 96000)                                                             # soft flat tops the app shows at its default Sureness, mono and at 96 kHz (softMono and soft96 are read with 41 %: not shown)
for name, src in (('trSpread', 'dmany/m_erf_1_05'), ('trRank', 'dmany/m_tanh_1_02'), ('trSides', 'dmany/m_alg_4_01'), ('trRough', 'dall/board_dist_pedalboard_01'), ('trOneSoft', 'dall/clip_clip_ff_cubic_00')):
    if os.path.exists(T + src + '_wet.wav'): shutil.copy(T + src + '_wet.wav', D + name + '.wav'); print(name, 'copied', flush=True)
for name, k in (('trOneSoft', 8), ('trSpread', 5)):      # the same several times in a row: the first pass works on parts, and what it tells sends the engine back (the first pass on the whole sound after all; nothing rebuilt)
    if os.path.exists(D + name + '.wav'): x, sr = sf.read(D + name + '.wav', dtype='float64', always_2d=True); w(name + 'Long', np.tile(x, (k, 1)), sr)
