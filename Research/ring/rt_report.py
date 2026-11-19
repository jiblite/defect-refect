"""What the ring-mod checks say: clean clips (rt2_clean_*.jsonl) and every scoreboard sound (rt2_fams_*.jsonl)."""
import json, glob, numpy as np
R = dict(match=1.0, dip=10.0, offset=-20.0, win=0.3, holes=0.1)
def rows(pat): return [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f) if l.strip()]
def crit(r): return dict(match=r['match'] <= R['match'], dip=r['dip'] >= R['dip'], offset=r['offset'] <= R['offset'], win=r['win'] >= R['win'], holes=r['holes'] <= R['holes']) if 'win' in r else None
def line(r): return '%-34s f %8.2f rho %5.2f eps %.1e win %5.2f (ideal %.2f) match %5.2f dip %5.1f offset %6.1f holes %.3f | first win %.2f | %.1f s' % (r['name'], r['f'], r['rho'], r['eps'], r['win'], r['ideal'], r['match'], r['dip'], r['offset'], r['holes'], r['first_win'], r['secs'])
c = rows('rt2_clean_[0-9].jsonl')
if c:
    print('clean: %d clips; found %d; went past the first look %d; secs median %.1f max %.1f' % (len(c), sum(r['found'] for r in c), sum('win' in r for r in c), np.median([r['secs'] for r in c]), max(r['secs'] for r in c)))
    for r in c:
        if r['found']: print('  WRONG ' + line(r))
    for r in sorted([r for r in c if 'win' in r and not r['found']], key=lambda r: -r['win'])[:12]: print('  near  ' + line(r) + ' fails ' + ','.join(k for k, v in crit(r).items() if not v))
    fw = sorted([r['first_win'] for r in c if 'first_win' in r], reverse=True); print('  first wins, largest: ' + ' '.join('%.2f' % v for v in fw[:16]))
o = rows('rt2_fams_[0-9].jsonl')
if o:
    by = {}
    for r in o: by.setdefault(r['name'].split('/')[0], []).append(r)
    print('scoreboard sounds: %d; secs median %.1f max %.1f' % (len(o), np.median([r['secs'] for r in o]), max(r['secs'] for r in o)))
    for k in sorted(by):
        q = by[k]; print('  %-12s %3d sounds: found %2d, past the first look %2d' % (k, len(q), sum(r['found'] for r in q), sum('win' in r for r in q)))
        for r in q:
            if r['found'] and k != 'ringmod': print('    WRONG ' + line(r))
            if k == 'ringmod' and not r['found']: print('    MISSED ' + (line(r) if 'win' in r else r['name'] + ' first win %.2f' % r.get('first_win', -9)))
            if not r['found'] and k != 'ringmod' and 'win' in r and sum(not v for v in crit(r).values()) == 1: print('    near  ' + line(r) + ' fails ' + ','.join(kk for kk, v in crit(r).items() if not v))
