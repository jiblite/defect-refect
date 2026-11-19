# The sounds for memcheck.py's stages "new" and "tiny" (the engines of build 0.4.7: sides, ring, crush, stutter):
# long sounds through a crusher and through a stutter, sounds with both, files at 44.1 and 96 kHz, mono files, second
# sounds at another rate and of ten samples, and sounds an engine could trip over (no samples at all, one sample, less
# than a window of the stutter's table, silence, a standing value, two levels, a sound that is one loop all through,
# an event that the file's end cuts off, values that are not numbers).
#   python3 mknew.py      (after mklong.py: it reads /dev/shm/mvl/clean00.wav and clean04.wav)  writes /tmp/claude-0/v2/newfx/mc/
import sys, os, numpy as np, soundfile as sf
from scipy.signal import resample_poly
here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [here + '/../crush', here + '/../stutter', '/tmp/claude-0/v2/sweep']
import crush, stutter
from lab import load
D = '/tmp/claude-0/v2/newfx/mc/'; os.makedirs(D, exist_ok=True); SETS = '/tmp/claude-0/v2/bench/sets/'; rng = np.random.default_rng(47)
rd = lambda p: sf.read(p, dtype='float64', always_2d=True)[0]
def w(name, x, sr=48000, sub='FLOAT'): sf.write(D + name + '.wav', x, sr, subtype=sub); print('%-16s %6d Hz %9d samples  %s' % (name, sr, len(x), sub))
def events(n, sr, plan):
    """plan: (start in s, slice in s, plays); nothing beyond the sound's end unless asked for."""
    return [(int(t * sr), int(L * sr), int(t * sr) + int(L * sr) * k) for t, L, k in plan if int(t * sr) + int(L * sr) <= n]
soft = dict(kind='soft', found=True, mix=0.7, soft=0.3, step=[2.0 / 2 ** 7] * 2, offset=[0.0, 0.0], P=3.3, c=0.4)
bare = lambda bits, P: dict(kind='bare', found=True, step=[2.0 / 2 ** bits] * 2, offset=[0.0, 0.0], P=P, c=0.0)
long_, long2 = rd('/dev/shm/mvl/clean00.wav'), rd('/dev/shm/mvl/clean04.wav'); six = load(SETS + 'bitcrush/bit_swh_00/', 'A')['dry']; sixB = load(SETS + 'bitcrush/bit_swh_00/', 'B')['dry']
n6 = len(six); plan6 = [(0.5, 0.06, 4), (1.6, 0.11, 3), (2.7, 0.045, 8), (4.0, 0.25, 3)]

# long sounds (the crush engine reads its settings from the loudest 30 s; the stutter's table grows with the sound)
w('crushSoftLong', crush.soft_forward(long_, soft))
w('crushBareLong', crush.through(np.vstack([long2, long_]), bare(6, 4.0)))
nl = len(long_); w('stutLong', stutter.play(long_, events(nl, 48000, [(0.0, 0.1, 4), (3.0, 0.05, 6), (9.0, 1.5, 3), (20.0, 0.02, 30), (31.0, 0.4, 2), (40.0, 0.2, 5)]) + [(nl - 9600, 2400, nl)], True))

# both effects on one sound: the crusher first (a stutter after it leaves the values on the grid)
w('both', stutter.play(crush.through(six, bare(7, 2.5)), events(n6, 48000, plan6), False))
w('bothSoft', stutter.play(crush.soft_forward(six, soft), events(n6, 48000, plan6), True))

# files that are not at 48 kHz (both engines read them at their own rate), and mono files
x44 = resample_poly(six, 147, 160, axis=0); x96 = resample_poly(six, 2, 1, axis=0)
w('soft44', crush.soft_forward(x44, dict(soft, P=2.7)), 44100)
w('both44', stutter.play(crush.through(x44, bare(8, 3.0)), events(len(x44), 44100, plan6), False), 44100, 'PCM_24')
w('stut96', stutter.play(x96, events(len(x96), 96000, plan6), True), 96000, 'PCM_24')
w('crush96', crush.through(x96, bare(6, 5.0)), 96000, 'PCM_24')
w('stutMono', stutter.play(six[:, :1], events(n6, 48000, plan6), False)[:, 0])
w('crushMono', crush.through(six[:, :1], dict(bare(5, 2.0), step=[2.0 / 2 ** 5], offset=[0.0]))[:, 0], 48000, 'PCM_16')

# second sounds: at another rate, of ten samples, mono
w('sec44', resample_poly(sixB, 147, 160, axis=0), 44100, 'PCM_24'); w('secTen', sixB[24000:24010]); w('secMono', sixB[:, 0])

# sounds an engine could trip over
noise = lambda n: 0.3 * rng.standard_normal((n, 2)) * np.exp(-np.arange(n) / max(n, 1))[:, None]
for n in (0, 1, 2, 63, 64, 65, 200, 1000, 4799): w('t_n%d' % n, noise(n))
sec = 48000; t = np.arange(sec)
w('t_silence', np.zeros((sec, 2))); w('t_dc', np.full((sec, 2), 0.25)); click = np.zeros((sec, 2)); click[sec // 2] = 0.9; w('t_click', click)
w('t_square', np.repeat(0.5 * np.sign(np.sin(2 * np.pi * 441.0 * t / 48000.0 + 0.1))[:, None], 2, 1))
w('t_three', 0.5 * np.round(np.clip(0.6 * rng.standard_normal((sec, 2)), -1, 1)))
w('t_loop', np.tile(noise(4800), (30, 1))); w('t_loopLong', np.tile(six[48000:72000], (12, 1)))
w('t_stutEnd', stutter.play(six, events(n6, 48000, plan6[:2]) + [(n6 - 7000, 3000, n6 + 2000)], False))
w('t_stutAll', stutter.play(six, [(0, 12000, n6)], False))
w('t_clipped', np.clip(3.0 * six, -1.0, 1.0)); w('t_faint', 1e-30 * rng.standard_normal((sec, 2)))
bad = six.copy(); bad[1000:1010] = np.nan; bad[50000, 0] = np.inf; bad[60000, 1] = -np.inf; w('t_notNumbers', bad)
w('t_oneSide', np.column_stack([six[:, 0], np.zeros(n6)])); w('t_loud', 40.0 * six)
