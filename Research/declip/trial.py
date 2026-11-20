# The whole declipper (the app's code as it stands, through drbend) against build 0.5.1's way back, on a folder of cases
# (name_wet.wav, name_dry.wav). Score: 100 = the sound before the effect, 0 = untouched, below 0 = worse.
#   TAG=x [EXE=...] [FIRST=n] python3 d38_run.py dir part parts [word]
import sys, os, glob, subprocess, time, pickle
os.chdir('/tmp/claude-0/v2/four'); sys.path.insert(0, '/home/claude/defect-refect/Research/bend')
from common import *
import bfeat
D = sys.argv[1].rstrip('/') + '/'; part, parts = int(sys.argv[2]), int(sys.argv[3]); word = sys.argv[4] if len(sys.argv) > 4 else ''; exe = os.environ.get('EXE', '/tmp/claude-0/v2/declip/bin/drbend'); model = '/home/claude/defect-refect/Resources/bend.bin'; tagx = os.environ.get('TAG', 'x')
names = sorted(os.path.basename(f)[:-8] for f in glob.glob(D + '*_wet.wav') if word in os.path.basename(f))[part::parts]; rows = []
for name in names:
    wet = sf.read(D + name + '_wet.wav', dtype='float64', always_2d=True)[0]; dry = sf.read(D + name + '_dry.wav', dtype='float64', always_2d=True)[0]; S0 = score.Scorer(dry); dw = S0(wet); pk = bfeat.peak_of(wet)
    sc = lambda o: 100 * (1 - S0(o * (pk / max(bfeat.peak_of(o), 1e-30))) / max(dw, 1e-9)); row = dict(name=name, dw=dw)
    for tag, env in (('plain', dict(DR_BEND_PLAIN='1')), ('new', dict(DR_FLAT_FIRST=os.environ['FIRST']) if os.environ.get('FIRST') else {})):
        pre = '/dev/shm/x38_%s_%d_%s' % (tagx, part, tag); t0 = time.time(); r = subprocess.run([exe, model, D + name + '_wet.wav', '-', pre, '0'], capture_output=True, text=True, env=dict(os.environ, **env)); row['sec_' + tag] = time.time() - t0
        if r.returncode: row[tag] = None; print(name, tag, 'FAILED', r.stderr[:200], flush=True); continue
        out = np.fromfile(pre + '.out.f64').reshape(len(wet), 2); row[tag] = sc(out); w = r.stdout.splitlines()[0].split(); row['found_' + tag] = int(w[1]); row['logit'] = float(w[3]); row['sure_' + tag] = float(w[5]); row['top_' + tag] = float(w[7]); row['kind_' + tag] = w[10]
        fl = [l for l in r.stdout.splitlines() if l.startswith('flat:')]
        if fl: f = fl[0].split(); g = lambda k: f[f.index(k) + 1]; row.update(kinds=g('kinds'), did=int(g('did')), rebuilt=int(g('rebuilt')), second=int(g('second')), whole=int(g('whole')), test=float(g('test')), spread=float(g('spread')), rho=float(g('rho')), rough=float(g('rough')), share=float(g('share')))
        for k in ('.out.f64', '.feat.f64', '.bend.f64'):
            if os.path.exists(pre + k): os.remove(pre + k)
    rows.append(row)
    print('%-28s dist %5.2f logit %5.1f | 0.5.1 %s %6.1f | new %s %6.1f (%+6.1f) %s (%.1f s)' % (name, dw, row.get('logit', 0), 'found' if row.get('found_plain') else '     ', row['plain'], 'found' if row.get('found_new') else '     ', row['new'], row['new'] - row['plain'],
          ('%s rebuilt %d second %d whole %d test %.1f spread %.2f rho %+.2f rough %.2f' % (row['kinds'], row['rebuilt'], row['second'], row['whole'], row['test'], row['spread'], row['rho'], row['rough'])) if 'kinds' in row else '', row['sec_new']), flush=True)
pickle.dump(rows, open('/tmp/claude-0/v2/declip/data/d38_%s_%d.pkl' % (tagx, part), 'wb'))
