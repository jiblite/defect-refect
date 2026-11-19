"""What the stereo checks say, all sets in one print: generated cases (st7_*.jsonl), clean clips (st5_clean_*.jsonl), other families (st5_fams_*.jsonl)."""
import json, glob, os, sys
import numpy as np
SETS = '/tmp/claude-0/v2/bench/sets/'
RULE = dict(taps=8, reach=60, made=-20.0, covered=0.5, inside=-20.0, beyond=-10.0, share=0.9, side=-15.0)
def rows(pat):
    r = []
    for f in sorted(glob.glob(pat)):
        r += [json.loads(l) for l in open(f) if l.strip()]
    return r
def tp(v): return ' '.join('%+.2f@%d' % (x, i) for i, x in v[:4])
def crit(r):
    """which of the pair's criteria a row passes"""
    if 'taps' not in r: return None
    return dict(taps=r['taps'] <= RULE['taps'], reach=r['reach'] >= RULE['reach'], made=r['made'] >= RULE['made'], covered=r['covered'] >= RULE['covered'], inside=r['inside'] <= RULE['inside'], beyond=r['beyond'] <= RULE['beyond'])
def line(r):
    s = '%-24s' % r['name']
    if 'taps' in r: s += ' taps %d apart %d reach %d made %.1f covered %.2f inside %.1f beyond %.1f' % (r['taps'], r['apart'], r['reach'], r['made'], r['covered'], r['inside'], r['beyond'])
    if 'share' in r: s += ' | share %.2f side %.1f' % (r['share'], r['side_db'])
    if 'a' in r: s += ' | a %s | b %s' % (tp(r['a']), tp(r['b']))
    return s
g = rows('st7_[0-9].jsonl')
if g:
    print('%d generated cases' % len(g))
    for e in sorted(set(r['effect'] for r in g), key=lambda e: (not e.startswith('haas'), e)):
        q = [r for r in g if r['effect'] == e]; f = [r for r in q if r.get('found')]
        print('%-22s n %2d found haas %2d width %2d none %2d | out median %5.1f (best possible %5.1f) out4 %5.1f (%5.1f) | secs %.1f' % (e, len(q), sum(r.get('found') == 'haas' for r in q), sum(r.get('found') == 'width' for r in q), sum(not r.get('found') for r in q),
              np.median([r['out'] for r in f]) if f else 0, np.median([r['best'] for r in f]) if f else 0, np.median([r['out4'] for r in f]) if f else 0, np.median([r['best4'] for r in f]) if f else 0, np.median([r['secs'] for r in q])))
    print('\nnot found, found under the other name, or far under the best possible:')
    for r in g:
        want = 'haas' if r['effect'].startswith('haas') else 'width'
        if r['effect'] == 'wid_calf_base': continue
        if r.get('found') != want or r['out'] < r['best'] - 8: print('  ' + line(r) + ' | found %s %s | out %s best %.0f | dry side %.1f' % (r.get('found'), r.get('mode'), '%.0f' % r['out'] if 'out' in r else '-', r['best'], r['dry_side']))
    fp = [r for r in g if r.get('mode') == 'pair']; fc = [r for r in g if r.get('mode') == 'copy']
    if fp: print('\nfound pairs (%d): taps max %d, reach min %d, made min %.1f, covered min %.2f, inside max %.1f, beyond max %.1f' % (len(fp), max(r['taps'] for r in fp), min(r['reach'] for r in fp), min(r['made'] for r in fp), min(r['covered'] for r in fp), max(r['inside'] for r in fp), max(r['beyond'] for r in fp)))
    if fc: print('found copies (%d): share min %.2f side min %.1f' % (len(fc), min(r['share'] for r in fc), min(r['side_db'] for r in fc)))
    b = sorted(fp, key=lambda r: -r['beyond'])[:6]; print('the pairs nearest the beyond limit: ' + ', '.join('%s %.1f' % (r['name'], r['beyond']) for r in b))
c = rows('st5_clean_[0-9].jsonl')
if c:
    w = [r for r in c if 'taps' in r or 'share' in r]; cnt = {}
    for r in w: cnt[r.get('found')] = cnt.get(r.get('found'), 0) + 1
    print('\nclean: %d clips, %d with a side; found: %s secs median %.1f max %.1f' % (len(c), len(w), cnt, np.median([r['secs'] for r in w]), max(r['secs'] for r in w)))
    for r in w:
        if r.get('found'): print('  WRONG ' + line(r))
    for k in ('taps', 'reach', 'made', 'covered', 'inside', 'beyond'):
        q = [r for r in w if crit(r) and not crit(r)[k] and all(v for kk, v in crit(r).items() if kk != k)]
        print('  clean, all but %-8s: %d %s' % (k, len(q), ' '.join('%s(%s=%.1f)' % (r['name'], k, r[k]) for r in q)))
    q = [r for r in w if crit(r) and sum(not v for v in crit(r).values()) == 2]
    print('  clean, all but two: %d %s' % (len(q), ' '.join('%s(%s)' % (r['name'], ','.join('%s=%.1f' % (k, r[k]) for k, v in crit(r).items() if not v)) for r in q)))
    q = [r for r in w if r.get('share', 0) >= 0.8]; print('  clean copy: share >= 0.8: ' + ' '.join('%s %.2f side %.1f' % (r['name'], r['share'], r['side_db']) for r in q))
o = rows('st5_fams_[0-9].jsonl')
if o:
    fam = {}
    for d in glob.glob(SETS + '*/*/'): fam[os.path.basename(os.path.dirname(d))] = os.path.basename(os.path.dirname(os.path.dirname(d)))
    print('\nother families: %d sounds' % len(o)); by = {}
    for r in o: by.setdefault(fam.get(r['name'].split(':')[0], '?'), []).append(r)
    for k in sorted(by):
        q = by[k]; w = [r for r in q if 'taps' in r or 'share' in r]
        print('  %-12s %3d sounds, %3d with a side: haas %2d width %2d | copy share max %.2f' % (k, len(q), len(w), sum(r.get('found') == 'haas' for r in q), sum(r.get('found') == 'width' for r in q), max([r.get('share', 0) for r in q] or [0])))
        if k not in ('haas', 'width'):
            for r in q:
                if r.get('found'): print('    WRONG ' + line(r))
            for r in w:
                cr = crit(r)
                if cr and not r.get('found') and sum(not v for v in cr.values()) == 1: print('    near: ' + line(r) + ' fails ' + ','.join(kk for kk, v in cr.items() if not v))
