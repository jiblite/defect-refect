# The stutter engine's far left on the fresh cases (gen_stutter): the found sound with the repeats cut out (Amount 0)
# against the dry sound with the parts that the made events covered cut out and closed up. The same sound to the
# sample when every event is found right.   python3 so1.py run ; python3 so1.py show
import sys, os, json, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
from sf7 import made, G, HERE
def closed(x, events, n):
    keep = np.ones(n, bool)
    for s0, L, end in events: keep[s0 + L:min(end, n)] = False
    o = np.zeros_like(x[:n]); k = int(keep.sum()); o[:k] = x[:n][keep]; return o
if sys.argv[1] == 'run':
    meta = json.load(open(G + 'meta.json')); out = open(HERE + 'so1.jsonl', 'w')
    for nm in sorted(meta):
        m = meta[nm]; ya = truth.rd(G + nm + '_A.flac') / m['gain']; xa = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['A'] + '.wav'); xb = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['B'] + '.wav'); n = min(len(ya), len(xa)); ya = ya[:n]; xa = xa[:n]
        want = [(a, b, min(c, n)) for a, b, c in made(m, n) if a + b < n]
        tmp = tempfile.mkdtemp(dir='/dev/shm'); pa, pb, pre = tmp + '/a.wav', tmp + '/b.wav', tmp + '/o'; sf.write(pa, ya, SR, subtype='FLOAT'); sf.write(pb, xb, SR, subtype='FLOAT')
        r = subprocess.run(['/tmp/claude-0/drstutter', pa, pb, pre], capture_output=True, text=True); J = json.loads(r.stdout.strip().splitlines()[-1]); row = dict(name=nm, effect=m['effect'], found=J['found'], made=len(want), right=len(set(map(tuple, J['events'])) & set(want)), got=len(J['events']))
        if J['found']:
            ya32 = truth.rd(pa)[:n]                                    # the sound as the program read it (32-bit float)
            o0 = np.fromfile(pre + '.own0.f64').reshape(2, -1).T[:n]; ideal = closed(xa, want, n); same = closed(ya32, [tuple(e) for e in J['events']], n)
            row['own0_db'] = db(o0 - ideal, ideal); row['self_db'] = db(o0 - same, same); row['score'] = score.removed(ideal, ya, o0)[0]
            row['cut_s'] = float(sum(min(c, n) - a - b for a, b, c in want)) / SR
        for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
        os.rmdir(tmp); out.write(json.dumps(row) + '\n'); out.flush()
else:
    rows = [json.loads(l) for l in open(HERE + 'so1.jsonl') if l.strip()]
    for e in ('stu_sox', 'stu_ffmpeg', 'plain'):
        q = [r for r in rows if r['effect'] == e and r['found']]
        if not q: continue
        d = np.array([r['own0_db'] for r in q]); sc = np.array([r['score'] for r in q]); allr = [r for r in q if r['right'] == r['made'] == r['got']]
        print('%-11s %2d sounds with events found; every event right on %2d of them. Amount 0 against the dry sound closed up: exact (under -120 dB) on %d, median %.1f dB, worst %.1f dB; score median %.1f, least %.1f; of the %d with every event right: exact on %d, worst %.1f dB'
              % (e, len(q), len(allr), int((d < -120).sum()), np.median(d), d.max(), np.median(sc), sc.min(), len(allr), sum(r['own0_db'] < -120 for r in allr), max([r['own0_db'] for r in allr] or [0])))
        print('            the engine against its own events (must be exact): worst %.1f dB' % max(r['self_db'] for r in q))
