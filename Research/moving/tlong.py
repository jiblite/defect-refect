# The moving-tone engine (the app's code, through drmoving) on the long test sounds of mklong.py.
#   EXE=path/to/drmoving python3 tlong.py [name ...]
import sys, os, json, subprocess, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/sweep']
from lab import score
EXE = os.environ.get('EXE', '/tmp/claude-0/lvx/drmoving2'); BIN = '/home/claude/defect-refect/Resources/moving.bin'; D = '/dev/shm/mvl/'; SR = 48000
cases = json.load(open(D + 'list.json')); only = sys.argv[1:]; rows = []
def rd(n): return sf.read(D + n + '.wav', dtype='float64', always_2d=True)[0]
def db(a, b): return 10 * np.log10(max(float(((a - b) ** 2).sum()), 1e-300) / max(float((b ** 2).sum()), 1e-300))
for c in cases:
    if only and c['name'] not in only: continue
    a = D + (c['name'] if c['kind'] == 'clean' else c['name'] + '_wet') + '.wav'; b = D + c.get('other', c.get('dry', c['name'])) + '.wav'
    r = subprocess.run([EXE, 'both', BIN, a, b, D + 'oa.f64', D + 'ob.f64'], capture_output=True, text=True); J = json.loads([l for l in r.stdout.split('\n') if l.startswith('{')][0])
    shown = J['found'] and J['sure'] > 0.65; sec = c['sec']; inside = sum(e - s for s, e in J['spans']) * 1024 / SR
    s = '%-8s %-16s %5.1f s | read in %4.1f s | %-11s sure %.2f (say %.2f, size %5.2f dB, read %d times) %s| in %4.1f s of it, %d of %d stretches' % (c['name'], c.get('tool', ''), sec, J['took'], J['kind'] or 'nothing', J['sure'], J['say'], J['size'], J['reads'], 'SHOWN ' if shown else '      ', inside, len(J['spans']), len(J['named']))
    row = dict(name=c['name'], kind=c['kind'], tool=c.get('tool', ''), found=J['found'], fkind=J['kind'], sure=J['sure'], say=J['say'], size=J['size'], reads=J['reads'], took=J['took'], inside=inside, sec=sec, named=J['named'], spans=J['spans'])
    if J['found'] and c['kind'] != 'clean':
        wet = rd(c['name'] + '_wet'); dry = rd(c['dry']) * c['gain']; out = np.fromfile(D + 'oa.f64').reshape(2, -1).T
        if c['kind'] in ('ph', 'sw'):
            row['out'] = score.removed(dry, wet, out)[0]; o = rd(c['other']); truth = rd(c['name'] + '_truth'); onto = np.fromfile(D + 'ob.f64').reshape(2, -1).T; row['onto'] = score.removed(truth, o, onto)[0]
            s += ' | out %6.1f onto %6.1f' % (row['out'], row['onto'])
        else:
            h = int(c['from_s'] * SR); row['out'] = score.removed(dry[h:], wet[h:], out[h:])[0]; row['before'] = db(out[:h - SR], wet[:h - SR]); first = min([sp[0] for sp in J['spans']]) * 1024 / SR
            s += ' | effect from %.1f s, found from %.1f s | its half: out %6.1f | the half before it changed by %.0f dB' % (c['from_s'], first, row['out'], row['before'])
    rows.append(row); print(s, flush=True)
json.dump(rows, open('tlong_%s.json' % os.environ.get('TAG', 'a'), 'w'))
