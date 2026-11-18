# The chorus engine as the app has it (drchorus both) on the scoreboard's chorus cases and the extra cases of mkx.py:
# found or not, the chorus taken out (against the dry sound) and put onto the case's other sound (against its wet sound).
#   EXE=path/to/drchorus python3 cbench.py [out.json]
import sys, os, json, subprocess, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
from lab import cases, load, score
EXE = os.environ.get('EXE', '/tmp/claude-0/drbuild/drchorus'); TMP = '/dev/shm/chp/'; os.makedirs(TMP, exist_ok=True); rows = []
def rdx(n, w, k): return sf.read('/dev/shm/chx/%s_%s_%s.wav' % (n, w, k), dtype='float64', always_2d=True)[0]
todo = [(os.path.basename(d.rstrip('/')), w, None) for d in cases('chorus') for w in 'AB'] + [('x%02d' % i, w, 1) for i in range(6) for w in 'AB']
for name, w, x in todo:
    o = 'B' if w == 'A' else 'A'
    if x: c = dict(wet=rdx(name, w, 'wet'), dry=rdx(name, w, 'dry')); q = dict(wet=rdx(name, o, 'wet'), dry=rdx(name, o, 'dry'))
    else: d = [p for p in cases('chorus') if os.path.basename(p.rstrip('/')) == name][0]; c = load(d, w); q = load(d, o)
    sf.write(TMP + 'a.wav', c['wet'], 48000, subtype='FLOAT'); sf.write(TMP + 'b.wav', q['dry'], 48000, subtype='FLOAT')
    r = subprocess.run([EXE, 'both', TMP + 'a.wav', TMP + 'b.wav', TMP + 'oa.f64', TMP + 'ob.f64'], capture_output=True, text=True); J = json.loads(r.stdout.strip().splitlines()[-1]); row = dict(name=name, which=w, found=bool(J['found']), sure=J['sure'], out=0.0, onto=0.0)
    if J['found']:
        out = np.fromfile(TMP + 'oa.f64').reshape(2, -1).T; onto = np.fromfile(TMP + 'ob.f64').reshape(2, -1).T; row['out'] = score.removed(c['dry'], c['wet'], out)[0]; row['onto'] = score.removed(q['wet'], q['dry'], onto)[0]
        v = [v for p in J['pieces'] for s in p['voices'] for v in s]; row['voices'] = len(J['pieces'][0]['voices'][0]); row['ahead'] = any(u['ahead'] for u in v)
    rows.append(row); print('%-20s %s found %d sure %.2f | out %7.1f onto %7.1f%s' % (name, w, row['found'], row['sure'], row['out'], row['onto'], '  (copy louder than the sound)' if row.get('ahead') else ''), flush=True)
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else 'cbench.json', 'w'))
f = [r for r in rows if r['found'] and not r['name'].startswith('x')]; print('scoreboard: found %d of 16 | where found: out median %.1f, onto median %.1f, worse than nothing: %d' % (len(f), np.median([r['out'] for r in f]), np.median([r['onto'] for r in f]), sum(r['out'] < 0 for r in f)))
