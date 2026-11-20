# Memory check of the app's whole processing (drrender built with -fsanitize=address,undefined): every engine's group on
# short and long first sounds, alone and with a second sound that is shorter, as long and longer, the faders in the middle
# and at both ends.
#   python3 memcheck.py [jobs] [rest|room|long|new|tiny|soft|move|movefx|four]    writes /dev/shm/mc/<name>.log, prints one line per run
# Two stages side by side: give each a folder of its own (MC_DIR=/dev/shm/mc2/ in the environment).
# Another build of the render test: MC_EXE=/path/to/drrender_asan in the environment.
import sys, os, json, subprocess, numpy as np, soundfile as sf
from concurrent.futures import ThreadPoolExecutor
sys.path[:0] = ['/tmp/claude-0/v2/sweep']; from lab import load
EXE = os.environ.get('MC_EXE', '/tmp/claude-0/lvx/drrender_asan'); RES = '/home/claude/defect-refect/Resources'; D = os.environ.get('MC_DIR', '/dev/shm/mc/'); SETS = '/tmp/claude-0/v2/bench/sets/'
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
MOVE = ['tremolo.speed', 'autopan.speed', 'gatedrhythm.speed', 'flanger.speed', 'flanger.sweep', 'flanger.feedback', 'chorus.speed', 'chorus.depth', 'phaser.speed', 'filtersweep.speed', 'delay.time', 'delay.feedback', 'delay.bright', 'reverb.length', 'reverb.bright', 'reverb.width', 'ringmod.carrier', 'bitcrush.bits', 'bitcrush.down', 'stutter.slice', 'comb.delay', 'tonematch.detail', 'filter.cutoff', 'filter.resonance']
def w(name, x): p = D + name + '.wav'; sf.write(p, x, 48000, subtype='FLOAT'); return p
firsts = []
fb = {}
for r in map(json.loads, open('/tmp/claude-0/v2/bench/famboard_044a.jsonl')):
    if r['found']: fb.setdefault(r['fam'], r['name'])
for fam in ('tremolo', 'autopan', 'gatedrhythm', 'flanger', 'delay', 'reverb', 'chorus', 'filtersweep', 'phaser', 'haas', 'width'):
    if fam in fb: firsts.append((fam + '6', w(fam + '6', load(SETS + fam + '/' + fb[fam] + '/', 'A')['wet'])))
firsts.append(('phaserB6', w('phaserB6', load(SETS + 'phaser/pha_pedalboard_01/', 'B')['wet'])))
firsts += [('sweepLong', '/dev/shm/mvl/swL07_wet.wav'), ('sweepHalf', '/dev/shm/mvl/swH01_wet.wav'), ('phaserLong', '/dev/shm/mvl/phL04_wet.wav'), ('chorusLong', '/dev/shm/chl/longA_wet.wav'), ('chorusHalf', '/dev/shm/chl/halfC_wet.wav')]
seconds = [('-', '-'), ('short', '/dev/shm/chl/shortD_dry.wav'), ('six', w('six', load(SETS + 'filtersweep/swp_mda_00/', 'B')['dry'])), ('long', '/dev/shm/mvl/clean00.wav')]
sets = [('middle', []), ('low', [k + '=0' for k in MOVE]), ('high', [k + '=2' for k in MOVE]), ('off', [k + '.amount=0' for k in ALL]), ('double', [k + '.amount=2' for k in ALL] + [k + '=2' for k in MOVE]), ('mixed', [k + '.amount=0.5' for k in ALL] + [k + '=0.37' for k in MOVE])]
def one(job):
    name, first, sure = job[:3]; args = []; i = 0; snd = job[3] if len(job) > 3 else seconds; sets = job[4] if len(job) > 4 else globals()['sets']      # a job may bring its own second sounds and fader sets
    if sure == -1: snd = [('tiny', first)]; first = '/tmp/claude-0/v2/newfx/mc/both.wav'; sure = None      # the tiny sound as the second one, under a first sound with a crusher and a stutter
    for sn, sp in snd:
        for fn, fad in sets:
            args += (['--'] if i else []) + [sp, D + 'out_%s.wav' % name] + fad + ([] if sure is None else ['sure=%d' % sure]); i += 1
    r = subprocess.run([EXE, RES, first] + args, capture_output=True, text=True, env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='print_stacktrace=1'))
    open(D + name + '.log', 'w').write(r.stdout + '\n--- stderr\n' + r.stderr)
    bad = [l for l in r.stderr.split('\n') if 'ERROR' in l or 'runtime error' in l]; shown = [l.split()[0] for l in r.stdout.split('\n') if ' p=' in l[:30]]
    if r.returncode == 1 and not bad and name.split('_s5')[0].endswith('n0') and r.stdout.strip().split('\n')[-1].startswith(('cannot read', 'no sound in')): return '%-16s refused, as the app refuses a file without sound | clean' % name
    return '%-16s exit %d | %d blocks | shown %s | %s' % (name, r.returncode, i, ','.join(sorted(set(shown)))[:80], ('PROBLEMS: ' + ' || '.join(sorted(set(b[:200] for b in bad))[:4])) if bad or r.returncode else 'clean')
jobs = [(n, p, None) for n, p in firsts] + [(n + '_s5', p, 5) for n, p in firsts if n in ('sweepLong', 'phaserLong', 'chorusHalf', 'flanger6', 'reverb6', 'phaser6', 'tremolo6')]
if 'rest' in sys.argv:           # the effect types without an engine of their own, the room and the tone engine: at the default and with Sureness at 5 (everything found is shown)
    import glob
    rest = [(fam + '6', w(fam + '6', load(sorted(glob.glob(SETS + fam + '/*/'))[0], 'A')['wet'])) for fam in ('reverb', 'tone', 'distortion', 'bitcrush', 'ringmod', 'stutter', 'exciter', 'saturation', 'filter')]
    jobs = [(n, p, None) for n, p in rest] + [(n + '_s5', p, 5) for n, p in rest]
if 'room' in sys.argv:           # sounds on which a room is found next to other effects, so that the look without those effects runs (Estimate.h, recheckRoomWithout): one for each combination of groups taken out
    W = '/tmp/claude-0/v2/roomsure/wav/nw/'; seconds = [seconds[0], seconds[2]]; sets = [sets[0], sets[3], sets[4]]
    jobs = [('room_' + n.replace('__', '_'), W + n + '.wav', None) for n in 'fxr__pan_mda__01 fx__cho_tap__03 fx__cho_pedalboard__03 rev__rev_dragonfly_hall__09 fxr__pan_sox__00 fxr__swp_pb_bandpass__00 fx__fil_sox__04 fx__swp_mda__03 fxr__cho_tap__01 fxr__gat_ffmpeg_keyed__00 fxr__haas_calf_stereotools__01 fx__haas_ffmpeg__04 fxr__haas_ffmpeg__02 fx__pha_calf__06 fx__rng_swh__08'.split()]
if 'long' in sys.argv:           # a sound of more than 9 s is judged in pieces (Estimate.h, roomInPieces): just over 9 s, silence around the sound, a gap in it, a short burst in silence (middle, end, start), more pieces than the most (78 s), and a room next to another effect (the look without it then runs in pieces too)
    W = '/tmp/claude-0/v2/roomsure/wav/nw/'; seconds = [seconds[0], seconds[2]]; sets = [sets[0], sets[3], sets[4]]
    rev = load(SETS + 'reverb/rev_calf_07/', 'A')['wet']; z = lambda sec: np.zeros((int(sec * 48000), 2)); rd = lambda p: sf.read(p, dtype='float64', always_2d=True)[0]; twice = lambda n: np.vstack([rd(W + n + '.wav')] * 2)
    made = [('just9', np.vstack([rev, z(0.1)])), ('silence', np.vstack([z(5), rev, z(8)])), ('gap', np.vstack([rev[:288000], z(7), rev[:288000]])), ('burst', np.vstack([z(10), rev[48000:96000], z(9)])),
            ('end', np.vstack([z(20), rev[48000:72000]])), ('start', np.vstack([rev[48000:72000], z(20)])), ('many', np.vstack([rd('/dev/shm/mvl/clean00.wav'), rd('/dev/shm/mvl/clean04.wav')])),
            ('pan', twice('fxr__pan_mda__01')), ('chorus', twice('fxr__cho_tap__01')), ('haas', twice('fxr__haas_ffmpeg__02')), ('echo', twice('rev__rev_dragonfly_hall__09'))]
    jobs = [('long_' + n, w('long_' + n, x), None) for n, x in made]
if 'new' in sys.argv:            # the engines of build 0.4.7 (sides, ring, crush, stutter; sounds from mknew.py): every kind of crusher, stutters, long sounds, both on one sound, files at 44.1 and 96 kHz, mono files; second sounds also at another rate and of ten samples
    N = '/tmp/claude-0/v2/newfx/mc/'; case = lambda fam, name, side='A': w(name, load(SETS + fam + '/' + name + '/', side)['wet'])
    seconds = [seconds[0], seconds[2], seconds[3], ('s44', N + 'sec44.wav'), ('ten', N + 'secTen.wav'), ('mono', N + 'secMono.wav')]
    new = [(n, case(f, n)) for f, n in (('ringmod', 'rng_swh_00'), ('haas', 'haas_ffmpeg_01'), ('haas', 'haas_mda_00'), ('width', 'wid_calf_haasenh_00'), ('width', 'wid_caps_01'),
                                         ('bitcrush', 'bit_calf_00'), ('bitcrush', 'bit_swh_00'), ('bitcrush', 'bit_pedalboard_01'), ('bitcrush', 'bit_mda_00'), ('stutter', 'stu_ffmpeg_00'), ('stutter', 'stu_sox_02'))]
    new += [(n, N + n + '.wav') for n in 'crushSoftLong crushBareLong stutLong both bothSoft soft44 both44 stut96 crush96 stutMono crushMono'.split()] + [('crush44', '/tmp/claude-0/v2/newfx/r44/crush_a44.wav'), ('stut44', '/tmp/claude-0/v2/newfx/r44/stut_a44.wav')]
    jobs = [(n, p, None) for n, p in new] + [(n + '_s5', p, 5) for n, p in new if n in ('rng_swh_00', 'haas_ffmpeg_01', 'wid_caps_01', 'bit_mda_00', 'stu_sox_02', 'both', 'bothSoft', 'both44')]
if 'soft' in sys.argv:           # a crusher that mixes plain sound in and is found (mknew.py: nine tenths crushed, a hold of three samples): at 44.1 and 96 kHz, where the sound before it is brought to 48 kHz like the sound itself, and on a long sound
    N = '/tmp/claude-0/v2/newfx/mc/'; seconds = [seconds[0], seconds[2], ('s44', N + 'sec44.wav')]
    jobs = [(n, N + n + '.wav', None) for n in 'softOk44 softOk96 softOkLong'.split()]
if 'tiny' in sys.argv:           # sounds an engine could trip over (mknew.py): no samples, one, less than a window, silence, a standing value, two levels, one loop all through, an event cut off by the file's end, values that are not numbers
    import glob
    N = '/tmp/claude-0/v2/newfx/mc/'; seconds = [seconds[0], seconds[2], ('ten', N + 'secTen.wav')]; sets = [sets[0], sets[3], sets[4]]
    tiny = [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(N + 't_*.wav'))]
    jobs = [(n, p, None) for n, p in tiny] + [(n + '_s5', p, 5) for n, p in tiny] + [('six_onto_' + os.path.basename(p)[2:-4], p, -1) for p in sorted(glob.glob(N + 't_n*.wav'))]
if 'move' in sys.argv:           # the MOVEMENT faders (build 0.5.0): on sounds that move, with hard cuts and gaps of silence, without a pitch, with found effects, at other rates, mono, long, and on the sounds an engine could trip over; alone and onto second sounds (the tiny ones and a long one among them)
    import glob
    N = '/tmp/claude-0/v2/newfx/mc/'; MV = ['movement.pitch', 'movement.loudness', 'movement.tone', 'movement.stereo', 'movement.spectral']; rng = np.random.default_rng(50); t = np.arange(6 * 48000) / 48000.0
    def tone(f0, bend): ph = 2 * np.pi * np.cumsum(f0 * 2 ** (bend / 1200.0)) / 48000.0; return sum(np.sin(k * ph + k) / k for k in range(1, 30))
    a = 0.1 * tone(110.0, 40 * np.sin(2 * np.pi * 5 * t) + 200 * t / 6) * (1 + 0.6 * np.sin(2 * np.pi * 0.4 * t)); pan = 0.5 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
    moving = w('mvTone', np.stack([a * (1 - pan), a * pan + 0.002 * rng.standard_normal(len(t))], 1))              # a vibrato on a glide, a swell, a pan sweep
    b = 0.1 * tone(180.0, 0 * t); g = np.stack([b + 0.02 * rng.standard_normal(len(t)), 0.7 * np.roll(b, 100) + 0.02 * rng.standard_normal(len(t))], 1)
    g[48000:48000 + 960] = 0; g[2 * 48000:2 * 48000 + 9600] = 0; g[3 * 48000:3 * 48000 + 4800, 0] = 0; g[4 * 48000:4 * 48000 + 300] = 0; g[5 * 48000:, 0] = 0; g[5 * 48000 + 700:, 1] = 0
    g[5 * 48000 + 5000:5 * 48000 + 5600] = 0.1 * rng.standard_normal((600, 2)); g[:2400] *= (np.linspace(0, 1, 2400) ** 4)[:, None]
    gaps = w('mvGaps', g)                                                                                           # takes with digital silence between them, a cut on one side only, a blip shorter than a frame, a quiet start
    hiss = w('mvNoise', 0.05 * rng.standard_normal((6 * 48000, 2)))                                                 # nothing whose pitch could be followed
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'movement')); import test_flow; TC = test_flow.cases()
    zaps = [('zaps%d' % i, w('mvZaps%d' % i, np.stack([TC[n][0]] * 2, 1))) for i, n in enumerate(('zaps apart', 'zaps back to back', 'zaps overlapping'))]       # no sustained sound: pieces whose place the reader has to prove, or leaves out
    ends = [('flat', [k + '=0' for k in MV]), ('double', [k + '=2' for k in MV])]
    msets = ends + [('mixed', ['movement.pitch=0.5', 'movement.loudness=1.5', 'movement.tone=0.37', 'movement.stereo=1.63', 'movement.spectral=0.2'] + [k + '.amount=0.5' for k in ALL]),
                    ('withFound', [k + '=2' for k in MV] + [k + '.amount=2' for k in ALL] + [k + '=2' for k in MOVE]), ('pitchOnly', ['movement.pitch=0.5'])]
    alone = [('-', '-')]; msec = [alone[0], seconds[2], ('ten', N + 'secTen.wav'), ('mono', N + 'secMono.wav'), ('s44', N + 'sec44.wav'), ('gaps', gaps), ('noise', hiss)]
    jobs = [('mv_' + n, q, None, msec, msets) for n, q in (('tone', moving), ('gaps', gaps), ('noise', hiss), ('both', N + 'both.wav'), ('both44', N + 'both44.wav'), ('stut96', N + 'stut96.wav'), ('crushMono', N + 'crushMono.wav'))]
    jobs += [('mv_both_s5', N + 'both.wav', 5, [alone[0], seconds[2]], msets), ('mv_long', '/dev/shm/mvl/clean00.wav', None, alone, msets), ('mv_ontoLong', moving, None, [('long', '/dev/shm/mvl/clean00.wav')], ends[:1] + msets[2:3])]
    jobs += [('mv_' + n, q, None, [alone[0], seconds[2]], ends) for n, q in zaps] + [('mv_onto_zaps', moving, None, zaps, ends)]
    tiny = sorted(glob.glob(N + 't_*.wav'))
    jobs += [('mv_' + os.path.basename(q)[:-4], q, None, alone, ends) for q in tiny] + [('mv_onto_' + os.path.basename(q)[2:-4], moving, None, [('tiny', q)], ends) for q in tiny]
if 'movefx' in sys.argv:         # the MOVEMENT faders next to every effect type that is found: where the first sound is measured without its level movement or its moving filter (Estimate.h, movementBase), and next to the others; alone and onto a second sound
    import glob
    MV = ['movement.pitch', 'movement.loudness', 'movement.tone', 'movement.stereo', 'movement.spectral']
    msets = [('flat', [k + '=0' for k in MV]), ('double', [k + '=2' for k in MV]), ('withFound', [k + '=2' for k in MV] + [k + '.amount=2' for k in ALL] + [k + '=2' for k in MOVE]),
             ('mixed', ['movement.pitch=0.5', 'movement.loudness=1.5', 'movement.tone=0.37', 'movement.stereo=1.63', 'movement.spectral=0.2'] + [k + '.amount=0.5' for k in ALL])]
    rest = [(fam + 'R', w(fam + 'R', load(sorted(glob.glob(SETS + fam + '/*/'))[0], 'A')['wet'])) for fam in ('tone', 'distortion', 'bitcrush', 'ringmod', 'stutter', 'exciter', 'saturation', 'filter')]
    jobs = [('mx_' + n, q, None, [seconds[0], seconds[2]], msets) for n, q in [f for f in firsts if f[0].endswith('6')] + rest] + [('mx_sweepHalf', '/dev/shm/mvl/swH01_wet.wav', None, [seconds[0]], msets[:2])]
if 'four' in sys.argv:           # the engines of build 0.5.1 (filter, bend; sounds from Research/filter/mkcheck.py): outside cases of each kind, long sounds, sounds at the limits of what is read, files at 44.1, 96 and 192 kHz, mono files, both on one sound; alone and onto second sounds, also with the MOVEMENT faders
    N = '/tmp/claude-0/v2/newfx/mc/'; Q = '/tmp/claude-0/v2/four/mc/'; case = lambda fam, name, side='A': w(name, load(SETS + fam + '/' + name + '/', side)['wet'])
    MV = ['movement.pitch', 'movement.loudness', 'movement.tone', 'movement.stereo', 'movement.spectral']
    seconds = [seconds[0], seconds[2], seconds[3], ('s44', N + 'sec44.wav'), ('ten', N + 'secTen.wav'), ('mono', N + 'secMono.wav')]
    sets = sets + [('withMoves', [k + '=1.6' for k in MV] + [k + '.amount=0' for k in ALL]), ('odd', ['filter.amount=0.3', 'filter.cutoff=1.9', 'filter.resonance=0', 'distortion.amount=0.01', 'saturation.amount=1.99'])]
    four = [(n, case(f, n)) for f, n in (('filter', 'fil_calf_00'), ('filter', 'fil_sox_00'), ('filter', 'fil_pb_ladder_00'), ('filter', 'fil_swh_ls_01'), ('distortion', 'dist_sox_00'), ('distortion', 'dist_pedalboard_01'),
                                          ('distortion', 'dist_caps_00'), ('saturation', 'sat_calf_tape_00'), ('saturation', 'sat_tap_tube_01'), ('saturation', 'sat_swh_valve_00'))]
    four += [(n, Q + n + '.wav') for n in 'filtLong filt1s filt0s9 filt44 filtMono bendLong bendShort bendHalf bend44 bend96 bend192 bend44pcm bendMono both bothOther'.split()]
    jobs = [(n, p, None) for n, p in four] + [(n + '_s5', p, 5) for n, p in four if n in ('fil_calf_00', 'fil_swh_ls_01', 'dist_caps_00', 'sat_swh_valve_00', 'filt1s', 'bendHalf', 'bend192', 'both', 'bothOther')]
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
    for line in ex.map(one, jobs): print(line, flush=True)
