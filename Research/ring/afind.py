# The engines of build 0.4.7 with the app's own code (drsides, drring: Tests/sides.cpp, Tests/ring.cpp) on every clean
# clip and on every scoreboard sound: what each of them claims.
#   python3 afind.py clean i n   -> afind_clean_<i>.jsonl      python3 afind.py fams i n   -> afind_fams_<i>.jsonl
#   python3 afind.py show        prints both
import sys, json, time, os, glob, subprocess, tempfile
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); SETS = '/tmp/claude-0/v2/bench/sets/'; BIN = os.environ.get('BIN', '/tmp/claude-0/')
TOOLS = [t for t in [('sides', 'drsides'), ('ring', 'drring')] if os.environ.get('ONLY', t[0]) == t[0]]
def mask(nm): return nm if not nm.startswith('real_') else 'one of the recordings'
def run(tool, path):
    r = subprocess.run([BIN + tool, 'find', path], capture_output=True, text=True)
    if r.returncode: return dict(error=r.stderr[-200:])
    return json.loads(r.stdout.strip().splitlines()[-1])
if sys.argv[1] in ('clean', 'fams'):
    i0, n0 = int(sys.argv[2]), int(sys.argv[3])
    if sys.argv[1] == 'clean':
        meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); items = [(nm, '/tmp/claude-0/v2/corpus/dry/' + nm + '.wav') for nm in sorted(meta)]
    else:
        items = [(os.path.basename(os.path.dirname(os.path.dirname(d))) + '/' + os.path.basename(os.path.dirname(d)) + ':' + w, d + w + '_wet.flac') for d in sorted(glob.glob(SETS + '*/*/')) if '/reverb/' not in d for w in 'AB']
    out = open(os.path.join(HERE, 'afind_%s_%d.jsonl' % (sys.argv[1], i0)), 'w'); tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None)
    for nm, path in items[i0::n0]:
        p = path
        if not path.endswith('.wav'): x, sr = sf.read(path, dtype='float64', always_2d=True); p = os.path.join(tmp, 'a.wav'); sf.write(p, x, sr, subtype='FLOAT')
        row = dict(name=nm)
        for key, tool in TOOLS: row[key] = run(tool, p)
        out.write(json.dumps(row) + '\n'); out.flush()
    sys.exit(0)
def rows(pat): return [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, pat))) for l in open(f) if l.strip()]
def said(r):
    s = []
    if r.get('sides', {}).get('kind'): s.append(r['sides']['kind'])
    if r.get('ring', {}).get('found'): s.append('ringmod')
    return s
c = rows('afind_clean_[0-9].jsonl')
if c:
    for key in ('sides', 'ring'):
        q = [r[key] for r in c if key in r]
        if q: print('clean: %d clips; %s: found on %d; seconds a clip: median %.2f max %.2f' % (len(q), key, sum(bool(r.get('kind') or r.get('found') is True) for r in q), np.median([r['took'] for r in q]), max(r['took'] for r in q)))
    for r in c:
        if said(r): print('  WRONG %s: %s' % (mask(r['name']), ', '.join(said(r))))
    if any('ring' in r for r in c):
        fw = sorted([r['ring'].get('first_win', -9) for r in c if r['ring'].get('looked')], reverse=True); print('  ring: a candidate line on %d clips; largest first wins: ' % len(fw) + ' '.join('%.2f' % v for v in fw[:8]) + '; went past the first look: %d' % sum(bool(r['ring'].get('measured')) for r in c))
        for r in c:
            if r['ring'].get('measured'): g = r['ring']; print('     past the first look: %s f %.2f rho %.2f win %.2f match %.2f dip %.1f offset %.1f holes %.3f' % (mask(r['name']), g['f'], g['rho'], g['win'], g['match'], g['dip'], g['offset'], g['holes']))
o = rows('afind_fams_[0-9].jsonl')
if o:
    by = {}
    for r in o: by.setdefault(r['name'].split('/')[0], []).append(r)
    own = dict(haas=('haas',), width=('width',), ringmod=('ringmod',)); print('scoreboard sounds: %d' % len(o))
    for k in sorted(by):
        q = by[k]; cnt = {}
        for r in q:
            for s in said(r): cnt[s] = cnt.get(s, 0) + 1
        print('  %-12s %3d sounds: %s' % (k, len(q), ', '.join('%s %d' % kv for kv in sorted(cnt.items())) or '-'))
        for r in q:
            w = [s for s in said(r) if s not in own.get(k, ())]
            if w: print('    WRONG %s: %s' % (r['name'], ', '.join(w)))
            g = r.get('ring', {})
            if g.get('measured') and not g.get('found') : print('    ring past the first look, not found: %s f %.2f rho %.2f win %.2f match %.2f dip %.1f offset %.1f holes %.3f' % (r['name'], g['f'], g['rho'], g['win'], g['match'], g['dip'], g['offset'], g['holes']))
