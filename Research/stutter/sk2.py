# Every clip of the corpus with a plain stutter put on (own draws: slices of 25 to 400 ms, 2 to 8 plays, with or without
# fades), through stutter.py or the app's own code: are the repeats read, is anything read that was not made, and how
# does the map do on another clip?   python3 sk2.py run [py|app] ; python3 sk2.py show
import sys, os, json, time, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import stutter as E, fc7
HERE = '/tmp/claude-0/v2/newfx/'
def cases():
    cm = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); clips = sorted(cm); out = []
    for i, clip in enumerate(clips):
        r = np.random.default_rng(1000 + i); out.append((clip, clips[(i + 37) % len(clips)], fc7.stutter_draw(r)))
    return out
if __name__ != '__main__': pass
elif sys.argv[1] == 'run':
    how = sys.argv[2] if len(sys.argv) > 2 else 'py'; out = open(HERE + 'sk2_%s.jsonl' % how, 'w')
    for k, (a, b, p) in enumerate(cases()):
        xa = truth.rd('/tmp/claude-0/v2/corpus/dry/' + a + '.wav'); xb = truth.rd('/tmp/claude-0/v2/corpus/dry/' + b + '.wav'); ya = fc7.plain_stutter(xa, p); yb = fc7.plain_stutter(xb, p); n = len(ya); want = [(s0, L, min(e, n)) for s0, L, e in p['events'] if s0 + L < n]; t0 = time.time()
        if how == 'py': f = E.find(ya); ev = [tuple(e) for e in f['events']]; found = f['found']; o = E.onto(xb, f) if found else None; fades = f.get('fades')
        else:
            tmp = tempfile.mkdtemp(dir='/dev/shm'); pa, pb, pre = tmp + '/a.wav', tmp + '/b.wav', tmp + '/o'; sf.write(pa, ya, SR, subtype='FLOAT'); sf.write(pb, xb, SR, subtype='FLOAT')
            r = subprocess.run(['/tmp/claude-0/drstutter', pa, pb, pre], capture_output=True, text=True); J = json.loads(r.stdout.strip().splitlines()[-1]); found = J['found']; ev = [tuple(e) for e in J['events']]; fades = J.get('fades')
            o = np.fromfile(pre + '.onto.f64').reshape(2, -1).T if found else None
            for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
            os.rmdir(tmp)
        dt = time.time() - t0; pk = float(np.abs(xa).max())
        def seen(e):                                                   # a repeat can be seen if what it repeats is not silence and does not stand still (a fifth of its samples differ from the one before), and it covers half a slice or 20 ms more than one play
            sl = ya[e[0]:e[0] + e[1]]; return bool(np.abs(sl).max() > 1e-3 * pk) and e[2] - e[0] - e[1] >= max(0.5 * e[1], 960) and int((np.abs(np.diff(sl, axis=0)) > E.TOL).any(1).sum()) >= max(0.2 * e[1], 256)
        near = lambda e, w: abs(e[0] - w[0]) <= 8 and e[1] == w[1]
        row = dict(k=k, A=a, B=b, fades_made=p['fades'], fades=fades, found=bool(found), made=len(want), seen=sum(seen(e) for e in want), got=len(ev), right=len(set(ev) & set(want)), took=dt,
                   extra=[e for e in ev if not any(near(e, w) for w in want)][:6], missed=[w for w in want if seen(w) and not any(near(e, w) for e in ev)][:6], other_end=[(e, w) for e in ev for w in want if near(e, w) and e != w][:6])
        if o is not None: nb = min(len(o), len(yb)); row['onto'] = score.removed(yb[:nb], xb[:nb], o[:nb])[0]; row['onto_db'] = db(o[:nb] - yb[:nb], yb[:nb])
        out.write(json.dumps(row) + '\n'); out.flush()
else:
    def mask(nm): return nm if not nm.startswith('real_') else 'one of the recordings'
    for how in ('py', 'app'):
        fn = HERE + 'sk2_%s.jsonl' % how
        if not os.path.exists(fn): continue
        rows = [json.loads(l) for l in open(fn) if l.strip()]; on = [r['onto'] for r in rows if 'onto' in r]
        print('%s: %d clips with a stutter put on: found on %d; repeats made %d, of them to be seen %d; read %d, right to the sample %d, same start and slice but another end %d, not made %d, missed %d; fades read right on %d of %d; onto another clip: median %.1f, lower quarter %.1f, least %.1f; %.2f s a sound (most %.1f)' % (how, len(rows), sum(r['found'] for r in rows), sum(r['made'] for r in rows), sum(r['seen'] for r in rows), sum(r['got'] for r in rows), sum(r['right'] for r in rows), sum(len(r['other_end']) for r in rows), sum(len(r['extra']) for r in rows), sum(len(r['missed']) for r in rows), sum(r['fades'] == r['fades_made'] for r in rows if r['found']), sum(r['found'] for r in rows), np.median(on), np.percentile(on, 25), min(on), np.median([r['took'] for r in rows]), max(r['took'] for r in rows)))
        for r in rows:
            if r['extra'] or r['missed'] or r['other_end'] or (not r['found'] and r['seen']) or os.environ.get('ALL'): print('      %-26s found %d: made %d seen %d got %d right %d | not made %s | missed %s | other end %s | onto %.1f' % (mask(r['A']), r['found'], r['made'], r['seen'], r['got'], r['right'], r['extra'][:3], r['missed'][:3], r['other_end'][:2], r.get('onto', 0)))
