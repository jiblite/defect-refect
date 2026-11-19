# Fresh ring-mod cases for the app's own code (drring, Tests/ring.cpp): n draws for each of four outside effects
# (zoo_fx.py) and for a plain ring modulation, on any clip of the corpus.
#   python3 rt7.py make n     -> genr/<effect>_<i>.flac, genr/meta.json
#   python3 rt7.py run i n    -> rt7_<i>.jsonl (what is found, the tone against what the effect did, removal score against the dry clip)
#   python3 rt7.py show
import sys, os, json, hashlib, time, glob, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/newfx'); sys.path.insert(0, '/home/claude/defect-refect/Research/detect')
from st_common import *
G = '/tmp/claude-0/v2/newfx/genr/'; os.makedirs(G, exist_ok=True); BIN = os.environ.get('BIN', '/tmp/claude-0/drring')
NAMES = ['rng_calf', 'rng_sox', 'rng_swh', 'rng_mda', 'plain']
def plain(x, p):
    t = np.arange(len(x)); g = (p['rho'] + np.sin(2 * np.pi * p['f'] / SR * t + p['phase'])) / np.sqrt(p['rho'] ** 2 + 0.5); return x * g[:, None]
def plain_draw(r): return dict(f=float(np.exp(r.uniform(np.log(25.0), np.log(4000.0)))), rho=float(0.0 if r.random() < 0.5 else r.uniform(0.0, 1.6)), phase=float(r.uniform(0, 2 * np.pi)))
def truth_gain(dry, wet):
    """The gain wet / dry fitted with the dry sound known: f, a, b, phase."""
    x, y = dry.sum(1), wet.sum(1); n = len(x); t = np.arange(n) / SR; ok = np.abs(x) > 0.05 * np.abs(x).max(); g = y[ok] / x[ok]; tt = t[ok]
    G0 = np.zeros(n); G0[ok] = g - g.mean(); F = np.abs(np.fft.rfft(G0 * np.hanning(n), 4 * n)); fr = np.fft.rfftfreq(4 * n, 1 / SR); f0 = fr[int(np.argmax(F[20:]) + 20)]
    for it in range(4):
        fs = f0 + np.linspace(-1, 1, 41) * (0.05 / 3 ** it); errs = []
        for f in fs: A = np.stack([np.ones_like(tt), np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)], 1); co = np.linalg.lstsq(A, g, rcond=None)[0]; errs.append(((A @ co - g) ** 2).sum())
        f0 = fs[int(np.argmin(errs))]
    A = np.stack([np.ones_like(tt), np.sin(2 * np.pi * f0 * tt), np.cos(2 * np.pi * f0 * tt)], 1); co = np.linalg.lstsq(A, g, rcond=None)[0]; res = A @ co - g
    return float(f0), float(co[0]), float(np.hypot(co[1], co[2])), float(np.arctan2(co[2], co[1])), float(np.sqrt(np.median(res ** 2)))
if sys.argv[1] == 'make':
    import zoo_fx as Z
    cm = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); clips = sorted(cm); meta = {}
    for name in NAMES:
        if name == 'plain': fam, sampler, clipf = 'ringmod', plain_draw, None
        else: fam, split, sampler, fn, clipf = Z.ZOO[name]
        for i in range(int(sys.argv[2])):
            rng = np.random.default_rng(int(hashlib.md5((name + 'genr/%d' % i).encode()).hexdigest(), 16) % 2 ** 32)
            for tries in range(8):
                p = sampler(rng); a = str(rng.choice(clips)); x = truth.rd('/tmp/claude-0/v2/corpus/dry/' + a + '.wav')
                if clipf is not None and not clipf(x): continue
                try: w = plain(x, p) if name == 'plain' else Z.apply(name, x, p, 0.0)
                except Exception as e: print(name, i, 'failed', str(e)[:100]); continue
                if np.isfinite(w).all() and np.abs(w).max() > 1e-3: break
            else: print(name, i, 'no case'); continue
            g = 1.0 if np.abs(w).max() < 0.98 else 0.98 / np.abs(w).max(); sf.write(G + '%s_%02d.flac' % (name, i), w * g, SR, format='FLAC', subtype='PCM_24')
            meta['%s_%02d' % (name, i)] = dict(effect=name, family=fam, clip=a, params=p, gain=g); print(name, i, a if not a.startswith('real_') else 'real_XX', flush=True)
    json.dump(meta, open(G + 'meta.json', 'w'), indent=1)
elif sys.argv[1] == 'run':
    meta = json.load(open(G + 'meta.json')); i0, n0 = int(sys.argv[2]), int(sys.argv[3]); out = open('/tmp/claude-0/v2/newfx/rt7_%d.jsonl' % i0, 'w'); tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None)
    for nm in sorted(meta)[i0::n0]:
        m = meta[nm]; wet = truth.rd(G + nm + '.flac') / m['gain']; dry = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['clip'] + '.wav'); n = min(len(wet), len(dry)); wet, dry = wet[:n], dry[:n]
        pa = os.path.join(tmp, 'a.wav'); sf.write(pa, wet, SR, subtype='FLOAT'); pre = os.path.join(tmp, 'o')
        r = subprocess.run([BIN, pa, pa, pre], capture_output=True, text=True); row = dict(name=nm, effect=m['effect'], clip=m['clip'], params=m['params'])
        if r.returncode: row['error'] = r.stderr[-200:]
        else:
            J = json.loads(r.stdout.strip().splitlines()[-1]); row.update(J); f0, a, b, ph, res = truth_gain(dry, wet); row.update(true_f=f0, true_rho=a / max(b, 1e-12), true_res=res)
            if J['found']:
                o = np.fromfile(pre + '.out.f64').reshape(2, -1).T; row['out'] = score.removed(dry, wet, o)[0]; row['out4'] = removed4(dry, wet, o)[0]
                row['out_db'] = db(o - dry, dry)
        out.write(json.dumps(row) + '\n'); out.flush()
else:
    rows = [json.loads(l) for fn in sorted(glob.glob('/tmp/claude-0/v2/newfx/rt7_[0-9].jsonl')) for l in open(fn) if l.strip()]
    print('%d fresh ring cases' % len(rows))
    for e in NAMES:
        q = [r for r in rows if r['effect'] == e]; fd = [r for r in q if r.get('found')]
        if not q: continue
        print('  %-9s found %2d of %2d' % (e, len(fd), len(q)) + ('; pitch off by at most %.5f Hz (median %.6f); taken out: median %.1f, least %.1f' % (max(abs(r['f'] - r['true_f']) for r in fd), np.median([abs(r['f'] - r['true_f']) for r in fd]), np.median([r['out'] for r in fd]), min(r['out'] for r in fd)) if fd else ''))
        for r in q:
            nm = r['clip'] if not r['clip'].startswith('real_') else 'one of the recordings'
            if not r.get('found'): print('      MISSED %-12s %-26s true f %8.2f rho %.3f | looked %s first win %.3f at %.2f Hz | measured %s f %.3f rho %.3f win %.2f match %.2f dip %.1f offset %.1f holes %.3f' % (r['name'], nm, r['true_f'], r['true_rho'], r.get('looked'), r.get('first_win', 0), r.get('first_f', 0), r.get('measured'), r.get('f', 0), r.get('rho', 0), r.get('win', 0), r.get('match', 0), r.get('dip', 0), r.get('offset', 0), r.get('holes', 0)))
            elif r['out'] < 90 or abs(r['f'] - r['true_f']) > 0.01: print('      low    %-12s %-26s true f %8.2f rho %.3f | f %.4f rho %.3f floor %.1e match %.2f dip %.1f | out %.1f (%.1f dB against the dry sound)' % (r['name'], nm, r['true_f'], r['true_rho'], r['f'], r['rho'], r['eps'], r['match'], r['dip'], r['out'], r['out_db']))
