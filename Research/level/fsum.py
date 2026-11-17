# summary of fast.py result files: python3 fsum.py file.jsonl [...] [--cases] [--base tag]
import sys, json, numpy as np
files = [a for a in sys.argv[1:] if not a.startswith('--') and a.endswith('.jsonl')]; base = sys.argv[sys.argv.index('--base') + 1] if '--base' in sys.argv else None
rows = [json.loads(l) for f in files for l in open(f)]; tags = list(rows[0]['v'].keys()); base = base or tags[0]
WANT = dict(tremolo='together', autopan='against', gatedrhythm='together')
def right(r, v): return v.get('found') and v['score'] >= 0 and v['kind'] == WANT[r['fam']]
def val(r, v, k): return v[k] if right(r, v) else 0.0
if '--cases' in sys.argv:
    for r in sorted(rows, key=lambda r: (r['mode'], r['i'])):
        b = r['v'][base]
        if all(abs(val(r, r['v'][t], 'out') - val(r, b, 'out')) < 0.5 and abs(val(r, r['v'][t], 'onto') - val(r, b, 'onto')) < 0.5 for t in tags): continue
        print('%-5s #%-3d %-16s %2d s %-14s' % (r['mode'], r['i'], r['effect'], r['seconds'], r['sound'][:14]), ' | '.join('%s %s%6.1f %6.1f%s' % (t, 's' if right(r, r['v'][t]) else ('l' if r['v'][t].get('found') else '-'), r['v'][t].get('out', 0), r['v'][t].get('onto', 0), ' (%.0f)' % r['v'][t]['out_off'] if 'out_off' in r['v'][t] else '') for t in tags))
for mode in sorted({r['mode'] for r in rows}):
    for fam in ('tremolo', 'gatedrhythm'):
        rs = [r for r in rows if r['mode'] == mode and r['fam'] == fam]
        if not rs: continue
        print('%-5s %-12s n %2d' % (mode, fam, len(rs)))
        for t in tags:
            o = np.array([val(r, r['v'][t], 'out') for r in rs]); q = np.array([val(r, r['v'][t], 'onto') for r in rs]); off = [r['v'][t]['out_off'] for r in rs if right(r, r['v'][t]) and 'out_off' in r['v'][t]]
            print('     %-4s shown %2d | out median %5.1f mean %5.1f worse %d | onto median %5.1f mean %5.1f worse %d%s' % (t, sum(right(r, r['v'][t]) for r in rs), np.median(o), o.mean(), (o < 0).sum(), np.median(q), q.mean(), (q < 0).sum(), ' | clean half touched (> -100 dB): %d of %d, worst %.0f dB' % (sum(v > -100 for v in off), len(off), max(off)) if off else ''))
