# The stutter engine on the fresh cases (fc7.py make stutter n): the events found against the events made, and the
# map put onto clip B against B through the same effect.   python3 sf7.py run [py|app] ; python3 sf7.py show
import sys, os, json, glob, time, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import stutter as E
G = '/tmp/claude-0/v2/newfx/gen_stutter/'; HERE = '/tmp/claude-0/v2/newfx/'
def plan(n, p):
    r = np.random.default_rng(p['seed']); L = int(p['slice_ms'] * 1e-3 * SR); pos = 0; out = []
    while pos + L <= n:
        if r.random() < p['prob']: out.append((pos, L, pos + p['repeats'] * L)); pos += p['repeats'] * L
        else: pos += L
    return out
def made(m, n): return [tuple(e) for e in m['params']['events']] if m['effect'] == 'plain' else plan(n, m['params'])
if __name__ != "__main__": pass
elif sys.argv[1] == "run":
    how = sys.argv[2] if len(sys.argv) > 2 else 'py'; meta = json.load(open(G + 'meta.json')); out = open(HERE + 'sf7_%s.jsonl' % how, 'w')
    for nm in sorted(meta):
        m = meta[nm]; ya = truth.rd(G + nm + '_A.flac') / m['gain']; yb = truth.rd(G + nm + '_B.flac') / m['gain']; xa = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['A'] + '.wav'); xb = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['B'] + '.wav'); n = min(len(ya), len(xa))
        want = [(a, b, min(c, n)) for a, b, c in made(m, n) if a + b < n]; t0 = time.time()
        if how == 'py': f = E.find(ya); ev = [tuple(e) for e in f['events']]; o = E.onto(xb, f) if f['found'] else None; found = f['found']
        else:
            tmp = tempfile.mkdtemp(dir='/dev/shm'); pa, pb, pre = tmp + '/a.wav', tmp + '/b.wav', tmp + '/o'; sf.write(pa, ya, SR, subtype='FLOAT'); sf.write(pb, xb, SR, subtype='FLOAT')
            r = subprocess.run(['/tmp/claude-0/drstutter', pa, pb, pre], capture_output=True, text=True); J = json.loads(r.stdout.strip().splitlines()[-1]); found = J['found']; ev = [tuple(e) for e in J['events']]
            o = np.fromfile(pre + '.onto.f64').reshape(2, -1).T if found else None
            for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
            os.rmdir(tmp)
        dt = time.time() - t0; live = lambda e: bool(np.abs(xa[e[0]:e[0] + e[1]]).max() > 1e-3) and e[2] - e[0] - e[1] >= 0.25 * e[1]
        row = dict(name=nm, effect=m['effect'], A=m['A'], B=m['B'], found=bool(found), made=len(want), seen=sum(live(e) for e in want), got=len(ev), right=len(set(ev) & set(want)), took=dt,
                   near=sum(1 for e in ev if e not in want and any(abs(e[0] - w[0]) <= 8 and e[1] == w[1] and abs(e[2] - w[2]) <= 8 for w in want)), extra=[e for e in ev if not any(abs(e[0] - w[0]) <= 8 and e[1] == w[1] for w in want)][:6], missed=[w for w in want if live(w) and not any(abs(e[0] - w[0]) <= 8 and e[1] == w[1] for e in ev)][:6])
        if o is not None: nb = min(len(o), len(yb)); row['onto'] = score.removed(yb[:nb], xb[:nb], o[:nb])[0]; row['onto_db'] = db(o[:nb] - yb[:nb], yb[:nb])
        out.write(json.dumps(row) + '\n'); out.flush()
else:
    def mask(nm): return nm if not nm.startswith('real_') else 'one of the recordings'
    for how in ('py', 'app'):
        fn = HERE + 'sf7_%s.jsonl' % how
        if not os.path.exists(fn): continue
        rows = [json.loads(l) for l in open(fn) if l.strip()]; print('%s: %d fresh stutter cases' % (how, len(rows)))
        for e in ('stu_sox', 'stu_ffmpeg', 'plain'):
            q = [r for r in rows if r['effect'] == e]
            if not q: continue
            on = [r['onto'] for r in q if 'onto' in r]
            print('  %-11s found %2d of %2d; events made %d, of them to be seen %d; found %d, right to the sample %d, within 8 samples %d more, not made %d, missed %d; onto B: median %.1f, least %.1f; %.2f s a sound' % (e, sum(r['found'] for r in q), len(q), sum(r['made'] for r in q), sum(r['seen'] for r in q), sum(r['got'] for r in q), sum(r['right'] for r in q), sum(r['near'] for r in q), sum(len(r['extra']) for r in q), sum(len(r['missed']) for r in q), np.median(on) if on else 0, min(on) if on else 0, np.median([r['took'] for r in q])))
            for r in q:
                if r['extra'] or r['missed'] or not r['found'] or r.get('onto', 100) < 90: print('      %-14s A %-24s B %-24s found %d: made %d seen %d got %d right %d | not made %s | missed %s | onto %.1f (%.1f dB)' % (r['name'], mask(r['A']), mask(r['B']), r['found'], r['made'], r['seen'], r['got'], r['right'], r['extra'][:3], r['missed'][:3], r.get('onto', 0), r.get('onto_db', 0)))
