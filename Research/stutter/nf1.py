# The new engines' references (crush.py, stutter.py) on every clean clip and on every scoreboard sound: what each claims.
#   python3 nf1.py clean|fams [crush|stutter]   -> nf1_<what>.jsonl ; python3 nf1.py show
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import crush as C, stutter as S
HERE = '/tmp/claude-0/v2/newfx/'
def claims(y):
    r = {}
    t0 = time.time(); f = C.find(y); r['crush'] = dict(found=f['found'], bits=f.get('bits'), P=f.get('P'), grids=[None if g is None else dict(step=g['step'], share=g['share'], levels=g['levels'], bits=g['bits']) for g in f.get('grids', [])], hold=f.get('hold'), took=time.time() - t0)
    t0 = time.time(); f = S.find(y); r['stutter'] = dict(found=f['found'], events=f['events'], whole=f.get('whole'), share=f.get('share'), took=time.time() - t0)
    return r
if sys.argv[1] in ('clean', 'fams'):
    if sys.argv[1] == 'clean':
        meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); items = [(nm, '/tmp/claude-0/v2/corpus/dry/' + nm + '.wav') for nm in sorted(meta)]
    else:
        items = [(os.path.basename(os.path.dirname(os.path.dirname(d))) + '/' + os.path.basename(os.path.dirname(d)) + ':' + w, d + w + '_wet.flac') for d in sorted(glob.glob(SETS + '*/*/')) if '/reverb/' not in d for w in 'AB']
    out = open(HERE + 'nf1_%s.jsonl' % sys.argv[1], 'w')
    for nm, path in items:
        y = truth.rd(path); row = dict(name=nm, seconds=len(y) / SR); row.update(claims(y)); out.write(json.dumps(row) + '\n'); out.flush()
else:
    def mask(nm): return nm if 'real_' not in nm else 'one of the recordings'
    for what in ('clean', 'fams'):
        fn = HERE + 'nf1_%s.jsonl' % what
        if not os.path.exists(fn): continue
        rows = [json.loads(l) for l in open(fn) if l.strip()]; print('%s: %d sounds' % (what, len(rows)))
        for eng in ('crush', 'stutter'):
            fam = dict(crush='bitcrush', stutter='stutter')[eng]; own = [r for r in rows if r['name'].startswith(fam + '/')]; oth = [r for r in rows if not r['name'].startswith(fam + '/')]
            print('   %s: found on %d of %d other sounds%s; seconds a sound: median %.2f, most %.2f' % (eng, sum(r[eng]['found'] for r in oth), len(oth), ('; on %d of its own %d' % (sum(r[eng]['found'] for r in own), len(own))) if own else '', np.median([r[eng]['took'] for r in rows]), max(r[eng]['took'] for r in rows)))
            for r in oth:
                if r[eng]['found']:
                    c = r[eng]; print('        %-34s %s' % (mask(r['name']), ('bits %.2f hold %.3f; grid shares %s levels %s' % (c['bits'], c['P'], ['%.4f' % g['share'] if g else '-' for g in c['grids']], [g['levels'] if g else 0 for g in c['grids']])) if eng == 'crush' else ('%d events, share %.2f: %s' % (len(c['events']), c['share'], c['events'][:4]))))
            if eng == 'stutter':
                for r in oth:
                    if r[eng].get('whole'): print('        (one slice over the whole sound, not counted: %s %s)' % (mask(r['name']), r[eng]['events'][:1]))
