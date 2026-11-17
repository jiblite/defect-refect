# The final long-sound files: the tremolo and gate rows of the rerun (long_tog_<new>.jsonl) laid over the run before
# (the pan rows stay as they were: the pan paths were not touched), the slow pans copied.   python3 merge.py
import json, os, shutil
def rd(p): return [json.loads(l) for l in open(p)]
for old, new, sets in (('p6', 'p7', ('wet', 'part', 'drift')), ('s1', 's2', ('wet', 'part'))):
    if not os.path.exists('long_tog_%s.jsonl' % new): print('no long_tog_%s.jsonl' % new); continue
    tog = {(r['mode'], r['i']): r for r in rd('long_tog_%s.jsonl' % new)}; used = 0
    for s in sets:
        rows = rd('long_%s_%s.jsonl' % (s, old)); n = 0; left = 0
        with open('long_%s_%s.jsonl' % (s, new), 'w') as f:
            for r in sorted(rows, key=lambda r: r['i']):
                k = (s, r['i'])
                if k in tog: r = tog[k]; n += 1
                elif r['fam'] != 'autopan': left += 1
                f.write(json.dumps(r) + '\n')
        used += n; print(new, s, len(rows), 'rows,', n, 'from the rerun,', left, 'tremolo or gate rows NOT rerun')
    print(new, 'rerun rows used', used, 'of', len(tog))
if os.path.exists('long_slow_p6.jsonl'): shutil.copy('long_slow_p6.jsonl', 'long_slow_p7.jsonl')
# the checks: the clean sets and the pan sample against the run before
def same(a, b, keys=('found', 'kind', 'score', 'out', 'onto')):
    return all((a.get(k) == b.get(k)) or (isinstance(a.get(k), float) and isinstance(b.get(k), float) and abs(a[k] - b[k]) < 0.05) for k in keys)
for old, new in (('p6', 'p7'), ('s1', 's2')):
    pa, pb = 'long_clean_%s.jsonl' % old, 'long_clean_%s.jsonl' % new
    if os.path.exists(pa) and os.path.exists(pb):
        A = {r['i']: r for r in rd(pa)}; B = {r['i']: r for r in rd(pb)}
        d = [i for i in sorted(A) if i not in B or not same(A[i], B[i], ('found', 'kind', 'score'))]
        print('clean', old, new, len(A), len(B), 'rows; different:', d)
if os.path.exists('long_pansample_p7.jsonl'):
    P = rd('long_pansample_p7.jsonl'); bad = []
    for r in P:
        o = next(q for q in rd('long_%s_p6.jsonl' % r['mode']) if q['i'] == r['i'])
        if not same(o, r): bad.append((r['mode'], r['i'], o.get('score'), r.get('score'), o.get('out'), r.get('out'), o.get('onto'), r.get('onto')))
    print('pan sample', len(P), 'rows; different:', bad)
