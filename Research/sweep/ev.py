# Blind scores on the bench cases of one family, both ways round; writes ev_<family>.json (per sound: scores and the decision's numbers).
#   python3 ev.py family [case substring] [--all: also score what the rule refuses]
import sys, os, json, time, numpy as np
sys.path.insert(0, '.'); from lab import *
import sweep
fam = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else ''; ALL = '--all' in sys.argv
rows = []
def num(f): return {k: (float(v) if not isinstance(v, str) else v) for k, v in f.items() if np.isscalar(v)}
def desc(f): return '-' if f is None else '%s d %.1f..%.1f ms g %+.2f fb %+.2f res %+.2f lev %+.2f/%+.2f seen %.2f' % (f['from'][0], f['dmin'] / 48, f['dmax'] / 48, f['g'], f['fb'], f['res'], f['level'], f['expect'], f['seen'])
for d in cases(fam):
    if sub not in d: continue
    cA = load(d, 'A'); cB = load(d, 'B')
    for c, o, tag in ((cA, cB, 'A'), (cB, cA, 'B')):
        t0 = time.time(); info = []; F = sweep.find(c['wet'], info=info); row = dict(name=c['name'], which=tag, split=c['meta']['split'], found=F is not None, out=0.0, onto=0.0, cands=[[num(f) for f in ch] for ch in info])
        if F is not None:
            out = sweep.apply(c['wet'], F, -1.0); onto = sweep.apply(o['dry'], F, 1.0)
            row['out'] = score.removed(c['dry'], c['wet'], out)[0]; row['onto'] = score.removed(o['wet'], o['dry'], onto)[0]
        if ALL:      # what each candidate would have scored, accepted or not (for judging the rule)
            for ch, cl in enumerate(info):
                for k, f in enumerate(cl):
                    if 'level' not in f or f['g'] == 0: continue
                    G = dict(kind='flanger', ch=[f if i == ch else None for i in range(2)]); out = sweep.apply(c['wet'], G, -1.0)
                    row['cands'][ch][k]['out_alone'] = score.removed(c['dry'][:, [ch, ch]], c['wet'][:, [ch, ch]], out[:, [ch, ch]])[0]; row['cands'][ch][k]['accepted'] = bool(sweep.accept(f))
        rows.append(row)
        print('%-20s %s %s out %7.1f onto %7.1f | %s  (%.1fs)' % (c['name'], tag, 'FOUND' if F else 'no   ', row['out'], row['onto'], ' | '.join(desc(f) for f in (F['ch'] if F else [])), time.time() - t0), flush=True)
        if ALL:
            for ch, cl in enumerate(row['cands']):
                for f in cl: print('      ch%d %-6s c1_first %+.2f g %+.2f fb %+.2f move %5.0f seen %.2f res %+.2f lev %+.2f/%+.2f -> %s out_alone %s' % (ch, f['from'], f['c1_first'], f['g'], f['fb'], f['move'], f['seen'], f['res'], f['level'], f['expect'], f.get('accepted'), '%.0f' % f['out_alone'] if 'out_alone' in f else '-'))
json.dump(rows, open('ev_%s%s.json' % (fam, sub), 'w'))
for sp in ('dev', 'test', None):
    v = np.array([(r['out'], r['onto'], r['found']) for r in rows if sp is None or r['split'] == sp])
    if len(v): print('%-4s found %d/%d | out: median %.1f mean %.1f worse %d | onto: median %.1f mean %.1f worse %d' % (sp or 'all', v[:, 2].sum(), len(v), np.median(v[:, 0]), v[:, 0].mean(), (v[:, 0] < 0).sum(), np.median(v[:, 1]), v[:, 1].mean(), (v[:, 1] < 0).sum()))
