# What the two filter engines say on one sound: the moving-tone engine (a sweep) and the fixed filter's reader, as the
# app's processing prints them. For the filter cases (scoreboard and fresh) and the scoreboard's sweeps, side A.
#   python3 both.py out.jsonl [jobs]
import sys, os, re, json, glob, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/tmp/claude-0/v2/proto'); import truth, soundfile as sf
EXE = os.environ.get('EXE', '/tmp/claude-0/v2/four/build/drrender_new'); RES = '/home/claude/defect-refect/Resources'; tmp = tempfile.mkdtemp()
cases = [('board', d) for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/filter/*/'))] + [('new', d) for d in sorted(glob.glob('/tmp/claude-0/v2/four/sets2/filter/*/'))] + [('board', d) for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/filtersweep/*/'))]
def one(c):
    tag, d = c; name = os.path.basename(d[:-1]); p = os.path.join(tmp, tag + name + '.wav'); sf.write(p, truth.load(d, 'A')['wet'], 48000, subtype='FLOAT')
    r = subprocess.run([EXE, RES, p, '-', os.path.join(tmp, tag + name + '_o.wav'), 'sure=5'], capture_output=True, text=True); o = r.stdout
    g = lambda pat: (re.search(pat, o, re.M) or [None, None])[1]
    row = dict(tag=tag, name=name, mov_found=g(r'^moving filter: found=(\d)'), mov_kind=g(r'^moving filter: .*kind=(\w+)'), mov_sure=g(r'^moving filter: .*? sure=([0-9.eE+-]+)'),
               fil_found=g(r'^filter: found=(\d)'), fil_sure=g(r'^filter: .*? sure=([0-9.eE+-]+)'), fil_logit=g(r'^filter: .*?logit=([0-9.eE+-]+)'), fil_kind=g(r'^filter: .*?kind=(\w+)'), shown=re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', o, re.M))
    os.remove(p); return row
with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as ex, open(sys.argv[1], 'w') as f:
    for row in ex.map(one, cases):
        f.write(json.dumps(row) + '\n'); f.flush()
        print('%-5s %-20s sweep: found %s kind %-11s sure %-9s | filter: found %s sure %-9s logit %-8s kind %-5s | shown with the slider down: %s' % (row['tag'], row['name'], row['mov_found'], row['mov_kind'], (row['mov_sure'] or '-')[:6], row['fil_found'], (row['fil_sure'] or '-')[:6], (row['fil_logit'] or '-')[:6], row['fil_kind'], ' '.join('%s %.2f' % (a, float(b)) for a, b in row['shown'])), flush=True)
