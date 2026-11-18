# Memory check of the app's whole processing (drrender built with -fsanitize=address,undefined): every engine's group on
# short and long first sounds, alone and with a second sound that is shorter, as long and longer, the faders in the middle
# and at both ends.
#   python3 memcheck.py [jobs] [rest|room|long]    writes /dev/shm/mc/<name>.log, prints one line per run
import sys, os, json, subprocess, numpy as np, soundfile as sf
from concurrent.futures import ThreadPoolExecutor
sys.path[:0] = ['/tmp/claude-0/v2/sweep']; from lab import load
EXE = '/tmp/claude-0/lvx/drrender_asan'; RES = '/home/claude/defect-refect/Resources'; D = '/dev/shm/mc/'; SETS = '/tmp/claude-0/v2/bench/sets/'
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
MOVE = ['tremolo.speed', 'autopan.speed', 'gatedrhythm.speed', 'flanger.speed', 'flanger.sweep', 'flanger.feedback', 'chorus.speed', 'chorus.depth', 'phaser.speed', 'filtersweep.speed', 'delay.time', 'delay.feedback', 'delay.bright', 'reverb.length', 'reverb.bright', 'reverb.width']
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
    name, first, sure = job; args = []; i = 0
    for sn, sp in seconds:
        for fn, fad in sets:
            args += (['--'] if i else []) + [sp, D + 'out_%s.wav' % name] + fad + ([] if sure is None else ['sure=%d' % sure]); i += 1
    r = subprocess.run([EXE, RES, first] + args, capture_output=True, text=True, env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='print_stacktrace=1'))
    open(D + name + '.log', 'w').write(r.stdout + '\n--- stderr\n' + r.stderr)
    bad = [l for l in r.stderr.split('\n') if 'ERROR' in l or 'runtime error' in l]; shown = [l.split()[0] for l in r.stdout.split('\n') if ' p=' in l[:30]]
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
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
    for line in ex.map(one, jobs): print(line, flush=True)
