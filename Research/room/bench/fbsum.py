# The summary rows of the README's scoreboard table from famboard result files: python3 fbsum.py a.jsonl [b.jsonl ...]
import sys, json, numpy as np
for p in sys.argv[1:]:
    rows = [json.loads(l) for l in open(p)]; n = len(rows)
    rev = sum('reverb' in r['others'] for r in rows); rest = sum(len([k for k in r['others'] if k != 'reverb']) for r in rows)
    w = lambda k: 100.0 * np.mean([r[k] < 0 for r in rows])
    print('%s: %d cases | own effect shown %d | other groups shown: REVERB %d / the rest %d | cases with no other group %d | worse than nothing: own far left %.0f %%, all far left %.0f %%, own onto %.0f %%, all onto %.0f %%'
          % (p.split('/')[-1], n, sum(r['found'] for r in rows), rev, rest, sum(not r['others'] for r in rows), w('out'), w('out_all'), w('onto'), w('onto_all')))
    for fam in ('tremolo', 'autopan', 'gatedrhythm', 'flanger'):
        rs = [r for r in rows if r['fam'] == fam]
        if rs: print('    %-12s shown %d/%d | out median %5.1f (where shown: %s) | onto median %5.1f (where shown: %s) | others: %s' % (fam, sum(r['found'] for r in rs), len(rs), np.median([r['out'] for r in rs]), ' '.join('%.0f' % r['out'] for r in rs if r['found']),
                     np.median([r['onto'] for r in rs]), ' '.join('%.0f' % r['onto'] for r in rs if r['found']), ' '.join(sorted({k for r in rs for k in r['others']}))))
