# The declipper on files at other rates and with one channel (mk_rates.py): build 0.5.1's way back against the new one.
#   python3 d39_rates.py dir
import sys, os, glob, subprocess, time
os.chdir('/tmp/claude-0/v2/four'); sys.path.insert(0, '/home/claude/defect-refect/Research/bend')
from common import *
import bfeat
D = sys.argv[1].rstrip('/') + '/'; exe = os.environ.get('EXE', '/tmp/claude-0/v2/declip/bin/drbend'); model = '/home/claude/defect-refect/Resources/bend.bin'
for f in sorted(glob.glob(D + '*_wet.wav')):
    name = os.path.basename(f)[:-8]; dry = sf.read(D + name + '_dry.wav', dtype='float64', always_2d=True)[0]; S0 = score.Scorer(dry); res = {}; line = ''
    for tag, env, amt in (('same', {}, '1'), ('plain', dict(DR_BEND_PLAIN='1'), '0'), ('new', {}, '0')):
        pre = '/dev/shm/x39_' + tag; t0 = time.time(); r = subprocess.run([exe, model, f, '-', pre, amt], capture_output=True, text=True, env=dict(os.environ, DR_BEND_ANY='1', **env)); sec = time.time() - t0
        if r.returncode: line += ' | %s FAILED %s' % (tag, r.stderr[:200]); continue
        o = np.fromfile(pre + '.out.f64').reshape(-1, 2); res[tag] = o
        if tag == 'new': fl = [l for l in r.stdout.splitlines() if l.startswith('flat:')]; line += ' | ' + r.stdout.splitlines()[0][:60] + ' | ' + (fl[0][:150] if fl else '') + ' (%.1f s)' % sec
    if len(res) == 3:
        n = min(len(dry), len(res['same'])); dw = S0(np.vstack([res['same'], np.zeros((max(0, len(dry) - len(res['same'])), 2))])); sc = lambda o: 100 * (1 - S0(np.vstack([o, np.zeros((max(0, len(dry) - len(o)), 2))])) / max(dw, 1e-9))
        line = '%-34s samples %d (clean %d) | 0.5.1 %6.1f new %6.1f' % (name, len(res['same']), len(dry), sc(res['plain']), sc(res['new'])) + line
    print(line, flush=True)
