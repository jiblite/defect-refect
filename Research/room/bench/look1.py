# First look: is a false REVERB on an effect sound the effect's doing, or is it already on the clean clip under it?
import json, collections, numpy as np
rows = json.load(open('sb_rows.json')); out = {}
for l in open('sb_out.jsonl'):
    j = json.loads(l); out[j['path']] = j
def verdict(j): return min(j['as']['sure'], j['end']['sure']) if j['as']['found'] else 0.09
for r in rows: r['j'] = out[r['path']]; r['v'] = verdict(r['j'])
clean = {r['dry']: r for r in rows if r['fam'] == 'clean'}
print('clean clips: shown %d of %d' % (sum(r['v'] > 0.65 for r in clean.values()), len(clean)))
kinds = collections.defaultdict(list)
for r in clean.values(): kinds[r['kind']].append(r['v'] > 0.65)
print('  by kind: ' + ', '.join('%s %d/%d' % (k, sum(v), len(v)) for k, v in sorted(kinds.items())))
tab = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for r in rows:
    if r['fam'] in ('clean',): continue
    c = clean[r['dry']]; t = tab[r['fam']]; t[0] += 1; t[1] += r['v'] > 0.65; t[2] += c['v'] > 0.65; t[3] += (r['v'] > 0.65) and (c['v'] > 0.65); t[4] += r['j']['as']['found']
print('%-12s %4s %6s %9s %5s %6s' % ('family', 'n', 'shown', 'cleanShn', 'both', 'found'))
tot = np.zeros(5, int)
for f, t in sorted(tab.items()):
    print('%-12s %4d %6d %9d %5d %6d' % (f, *t))
    if f not in ('reverb', 'tone'): tot += np.array(t)
print('%-12s %4d %6d %9d %5d %6d' % ('17 families', *tot))
# the A-direction cases the app's scoreboard marks
fb = [json.loads(l) for l in open('/tmp/claude-0/v2/bench/famboard_044a.jsonl')]
fa = {(r['fam'], r['name']): r for r in fb}
n = 0
for r in rows:
    if r['which'] != 'A' or r['fam'] in ('reverb', 'tone', 'clean'): continue
    k = fa.get((r['fam'], r['case']))
    app = k is not None and 'reverb' in str(k.get('others', ''))
    if app or r['v'] > 0.65:
        c = clean[r['dry']]; j = r['j']; n += 1
        print('  %-12s %-22s app %d | here %.2f (as %.2f end %.2f) | clean %s: %.2f (as %.2f end %.2f) | deep %.3f bands %2d share %5.1f decay %.2f | clean deep %.3f bands %2d' % (r['fam'], r['case'], app, r['v'], j['as']['sure'], j['end']['sure'], r['dry'], c['v'], c['j']['as']['sure'], c['j']['end']['sure'], j['as']['deep'], j['as']['bands'], j['as']['share'], j['as']['decay'], c['j']['as']['deep'], c['j']['as']['bands']))
print(n)
