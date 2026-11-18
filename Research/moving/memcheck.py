# Memory check of the app's whole processing (drrender built with -fsanitize=address,undefined): every engine's group on
# short and long first sounds, alone and with a second sound that is shorter, as long and longer, the faders in the middle
# and at both ends.
#   python3 memcheck.py [jobs] [rest]         writes /dev/shm/mc/<name>.log, prints one line per run
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
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
    for line in ex.map(one, jobs): print(line, flush=True)
