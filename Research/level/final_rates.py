# The final rule as it stands in level.py (rule.json + hard conditions), counted on the rule data: clean clips reported, movements found.
import json, sys, numpy as np, collections
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level
GATED = {'tr_synthetic_005', 'tr_synthetic_055'}
rows = [r for r in (json.loads(l) for l in open('/tmp/claude-0/v2/level/data/stage2.jsonl')) if r['name'] not in GATED]
dev = [json.loads(l) for l in open('/tmp/claude-0/v2/level/data/corpus_dev.jsonl')]
rep = [r['name'] for r in rows if any(level.accept_together(m) for m in r['clean'])]; repd = [r['name'] for r in dev if any(level.accept_together(m) for m in r['tog'])]
print('clean train clips reported (together): %d of %d %s; corpus dev: %d of %d %s' % (len(rep), len(rows), rep, len(repd), len(dev), repd))
got = collections.Counter(); tot = collections.Counter(); wrong = 0
for r in rows:
    p = r['par']
    if p['kind'] not in ('trem', 'gate'): continue
    ks = [p['kind'], p['kind'] + (' deep' if p['depth'] >= 0.6 else ' shallow')] + ([p['kind'] + ' ' + p['shape']] if 'shape' in p else []) + (['gate pattern' if p['pattern'] != 0xFFFF else 'gate plain'] if p['kind'] == 'gate' else [])
    for k in ks: tot[k] += 1
    best = level.choose_together(r['wet'])
    if best is None: continue
    if best['match'] in (1, 2, 3, 4):
        for k in ks: got[k] += 1
    else: wrong += 1
print('found (the chosen candidate has the true period or a whole multiple): ' + '  '.join('%s %d/%d (%.0f%%)' % (k, got[k], tot[k], 100 * got[k] / max(tot[k], 1)) for k in sorted(tot)) + ' | wrong period chosen: %d' % wrong)
pr = [json.loads(l) for l in open('/tmp/claude-0/v2/level/data/pan_train.jsonl')]
def right(m, par):
    r = (1000.0 / m['P']) / par['rate']; return bool(abs(r - 1) < 0.03 or abs(r * 2 - 1) < 0.03 or (m['slow'] and 0.6 < r < 1.6))
print('pan: clean train clips reported %d of %d %s; synthetic pans found with the right period %d of %d; corpus dev clean reported %d of %d' % (sum(1 for r in pr if r['clean'] and level.accept_pan(r['clean'])), len(pr), [r['name'] for r in pr if r['clean'] and level.accept_pan(r['clean'])],
      sum(1 for r in pr if r['wet'] and level.accept_pan(r['wet']) and right(r['wet'], r['par'])), len(pr), sum(1 for r in dev if r['pan'] and level.accept_pan(r['pan'])), len(dev)))
