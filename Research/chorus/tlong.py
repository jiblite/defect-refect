# The app's chorus engine on long, half-effected and short sounds (mklong.py): time, what it finds, and the scores.
#   python3 tlong.py     env EXE
import sys, os, json, subprocess, time, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
from lab import score
EXE = os.environ.get('EXE', '/tmp/claude-0/lvx/drchorus'); D = '/dev/shm/chl/'; rd = lambda p: sf.read(p, dtype='float64', always_2d=True)[0]
def raw(p, n): return np.fromfile(p).reshape(2, n).T
def run(a, b):
    for f in ('oa.f64', 'ob.f64'):
        if os.path.exists(D + f): os.remove(D + f)
    t0 = time.time(); r = subprocess.run([EXE, 'both', D + a + '_wet.wav', D + b + '_dry.wav', D + 'oa.f64', D + 'ob.f64'], capture_output=True, text=True); t1 = time.time()
    J = json.loads(r.stdout.strip().splitlines()[-1]); return J, t1 - t0
def show(J):
    for pc in J['pieces']:
        print('    piece from %5.1f s, %4.1f s long: %s' % (pc['a'] / 48000, pc['n'] / 48000, ' | '.join(', '.join('%s %.3f Hz %.2f+-%.2f ms on %.2f loud %.2f%s' % ('tri' if v['kind'] else 'sin', v['rate'], v['mid'] / 24, abs(v['half']) / 24, v['on'], v['loud'], ' LOUDER' if v['ahead'] else '') for v in side) or '-' for side in pc['voices'])))
    for n in J['notes'][:12]: print('      ' + n)
for a, b in (('longA', 'longB'), ('halfC', 'longB'), ('shortD', 'longB'), ('edgeE', 'shortD')):
    wet = rd(D + a + '_wet.wav'); dry = rd(D + a + '_dry.wav'); odry = rd(D + b + '_dry.wav'); owet = rd(D + b + '_wet.wav'); J, took = run(a, b)
    print('%s (%.1f s) with %s (%.1f s): %.1f s in all, find %.1f s | found %d sure %.2f' % (a, len(wet) / 48000, b, len(odry) / 48000, took, J['took'], J['found'], J['sure']), flush=True); show(J)
    if not J['found']: continue
    out = raw(D + 'oa.f64', len(wet)); onto = raw(D + 'ob.f64', len(odry)); seg = 6 * 48000
    print('    taken out: %.1f of 100 | by 6 s: %s' % (score.removed(dry, wet, out)[0], ' '.join('%.0f' % score.removed(dry[i:i + seg], wet[i:i + seg], out[i:i + seg])[0] for i in range(0, len(wet) - seg + 1, seg))), flush=True)
    if a == 'halfC': k = 14 * 48000; print('    the first 14 s (no chorus there): changed by %.1f dB against the sound' % (10 * np.log10(((out[:k] - wet[:k]) ** 2).mean() / (wet[:k] ** 2).mean() + 1e-30)), '| 14 to 17 s: %.1f dB' % (10 * np.log10(((out[k:17 * 48000] - wet[k:17 * 48000]) ** 2).mean() / (wet[k:17 * 48000] ** 2).mean() + 1e-30)))
    if a in ('longA', 'edgeE') and b != 'shortD': print('    put onto %s: %.1f of 100 | by 6 s: %s' % (b, score.removed(owet, odry, onto)[0], ' '.join('%.0f' % score.removed(owet[i:i + seg], odry[i:i + seg], onto[i:i + seg])[0] for i in range(0, len(odry) - seg + 1, seg))), flush=True)
