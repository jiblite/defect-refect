# A ring modulation and an echo on one sound, in both orders: what the ring engine and the echo engine say on the sound
# as it is, and what the echo engine says on the sound without the tone (the app's code: drrender, "echo without the tone").
#   python3 rc1.py i n -> rc1_<i>.jsonl      python3 rc1.py show
import sys, json, os, glob, subprocess, re
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); SR = 48000; BIN = os.environ.get('RENDER', '/tmp/claude-0/drrender047'); RES = '/home/claude/defect-refect/Resources'
def ring(x, f, rho, ph):
    t = np.arange(len(x)); g = (rho + np.sin(2 * np.pi * f / SR * t + ph)) / np.sqrt(rho * rho + 0.5); return x * g[:, None]
def echo(x, ms, a, fb=0.0):
    d = int(round(ms * SR / 1000)); y = x.copy(); w = a; k = 1
    while w > 1e-3 and k * d < len(x):
        y[k * d:] += w * x[:len(x) - k * d]; w *= fb; k += 1
        if fb == 0: break
    return y
def look(path):
    r = subprocess.run([BIN, RES, path, '-', '-'], capture_output=True, text=True); o = dict(ring=0, echo=None, without=None, counts=None, delay=None)
    for l in r.stdout.splitlines():
        if l.startswith('ring: found='): o['ring'] = int(l.split('found=')[1][0]); m = re.search(r' hz=([-0-9.e+]+)', l); o['hz'] = float(m.group(1)) if m else None
        elif l.startswith('echo: sure='): o['echo'] = float(l.split('sure=')[1].split()[0]); o['echo_line'] = l[:160]
        elif l.startswith('echo without the tone'): o['without'] = float(l.split('sureness=')[1].split()[0]); o['counts'] = float(l.split('counts=')[1].split()[0])
        elif re.match(r'^delay p=', l): o['delay'] = float(l.split('p=')[1].split()[0])
    return o
if sys.argv[1] != 'show':
    i0, n0 = int(sys.argv[2 - 1]), int(sys.argv[2]); meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); names = [nm for nm in sorted(meta) if not nm.startswith('real_')]
    rng = np.random.default_rng(7); pick = [names[i] for i in rng.permutation(len(names))[:16]]; tmp = '/tmp/claude-0/v2/newfx/rc1_tmp_%d' % i0; os.makedirs(tmp, exist_ok=True); out = open(os.path.join(HERE, 'rc1_%d.jsonl' % i0), 'w')
    jobs = []
    for k, nm in enumerate(pick):
        r2 = np.random.default_rng(100 + k); f = float(np.exp(r2.uniform(np.log(80), np.log(2500)))); ph = float(r2.uniform(0, 6.28)); ms = float(r2.choice([120, 180, 250, 375, 500]))
        jobs += [(nm, 'ring', f, ph, ms, 0.0), (nm, 'echo', f, ph, ms, 0.5), (nm, 'echo>ring', f, ph, ms, 0.5), (nm, 'echo>ring', f, ph, ms, 0.25), (nm, 'ring>echo', f, ph, ms, 0.15), (nm, 'ring>echo', f, ph, ms, 0.3), (nm, 'ring>echo', f, ph, ms, 0.5)]
    for nm, kind, f, ph, ms, a in jobs[i0::n0]:
        x, sr = sf.read('/tmp/claude-0/v2/corpus/dry/' + nm + '.wav', dtype='float64', always_2d=True); assert sr == SR
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        if kind == 'ring': y = ring(x, f, 0.0, ph)
        elif kind == 'echo': y = echo(x, ms, a, 0.4)
        elif kind == 'echo>ring': y = ring(echo(x, ms, a, 0.4), f, 0.0, ph)
        else: y = echo(ring(x, f, 0.0, ph), ms, a, 0.4)
        y = y * (0.5 / max(1e-9, np.abs(y).max())); p = os.path.join(tmp, 'a.wav'); sf.write(p, y, SR, subtype='FLOAT')
        row = dict(name=nm, kind=kind, f=f, ms=ms, a=a); row.update(look(p)); out.write(json.dumps(row) + '\n'); out.flush()
    sys.exit(0)
rows = [json.loads(l) for fn in sorted(glob.glob(os.path.join(HERE, 'rc1_[0-9].jsonl'))) for l in open(fn) if l.strip()]
CUT = 0.65; by = {}
for r in rows: by.setdefault((r['kind'], r['a']), []).append(r)
print('%d sounds; "shown" = the DELAY group at the default Sureness (%.2f)' % (len(rows), CUT))
for k in sorted(by):
    q = by[k]; fr = [r for r in q if r['ring']]; sh0 = sum((r['echo'] or 0) > CUT for r in q); sh1 = sum((r['counts'] if r['counts'] is not None else (r['echo'] or 0)) > CUT for r in q)
    print('  %-10s echo level %.2f: %2d sounds, ring found on %2d; delay shown before %2d, after %2d; looked at again %2d' % (k[0], k[1], len(q), len(fr), sh0, sh1, sum(r['without'] is not None for r in q)))
    for r in q:
        if r['without'] is not None: print('      %-22s %7.1f Hz %3.0f ms: on the sound %.3f, without the tone %.3f' % (r['name'] if not r['name'].startswith('real_') else 'one of the recordings', r['f'], r['ms'], r['echo'], r['without']))
