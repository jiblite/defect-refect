# The groups the app shows on every dry corpus sound (6 s clips with no effect on them), at the default Sureness.
#   python3 drycheck.py path/to/drrender tag part of        results: drycheck_<tag>_<part>.jsonl
#   python3 drycheck.py --show tag [tag ...]
import sys, os, json, subprocess, re, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); RES = '/home/claude/defect-refect/Resources'; CORPUS = '/tmp/claude-0/v2/corpus'
def show(tags):
    meta = json.load(open(CORPUS + '/meta.json'))
    for tag in tags:
        rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, 'drycheck_%s_*.jsonl' % tag))) for l in open(f)]
        for split in ('dev', 'test', 'all'):
            rs = [r for r in rows if split == 'all' or meta[r['name']]['split'] == split]; c = collections.Counter(k for r in rs for k in r['shown'])
            print('%-6s %-4s %3d sounds | nothing shown %3d | nothing but REVERB %3d | groups per sound %.2f (without REVERB %.2f) | %s' % (tag, split, len(rs), sum(not r['shown'] for r in rs),
                  sum(not [k for k in r['shown'] if k != 'reverb'] for r in rs), sum(len(r['shown']) for r in rs) / max(1, len(rs)), sum(len([k for k in r['shown'] if k != 'reverb']) for r in rs) / max(1, len(rs)),
                  ' '.join('%s %d' % kv for kv in c.most_common())))
if sys.argv[1] == '--show': show(sys.argv[2:]); sys.exit(0)
EXE, tag, part, of = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); out = os.path.join(HERE, 'drycheck_%s_%d.jsonl' % (tag, part)); tmp = '/dev/shm/dc/%s_%d.wav' % (tag, part)
done = {json.loads(l)['name'] for l in open(out)} if os.path.exists(out) else set()
for f in sorted(glob.glob(CORPUS + '/dry/*.wav'))[part::of]:
    n = os.path.basename(f)[:-4]
    if n in done: continue
    r = subprocess.run([EXE, RES, f, '-', tmp], capture_output=True, text=True); o = r.stdout
    if r.returncode: print('FAILED', n, r.stderr[-200:], flush=True); continue
    sh = re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', o, re.M); lev = re.search(r'^level: (.*)$', o, re.M); flg = re.search(r'^flanger: (.*)$', o, re.M)
    open(out, 'a').write(json.dumps(dict(name=n, shown=dict((k, float(v)) for k, v in sh), level=lev.group(1) if lev else '', flanger=flg.group(1) if flg else '')) + '\n')
