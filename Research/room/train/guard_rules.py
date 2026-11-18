# Rules w(excess, mid bands) for the guard, read off guard_bands2_<FIRST>.json.
import sys, json, numpy as np
FIRST = sys.argv[1] if len(sys.argv) > 1 else 'unet_y1'; R = json.load(open('/tmp/claude-0/v2/train/guard_bands2_%s.json' % FIRST)); WS = np.arange(11) / 10.0
def at(r, w): return float(np.interp(w, WS, r['line']))
def lin(D): return lambda e, m: 1.0 if e <= D else D / e
def quad(D): return lambda e, m: 1.0 if e <= D else (D / e) ** 2
def byev(Dthin, Drich, m0=2, m1=6, p=1):
    def f(e, m): D = Dthin if m <= m0 else Drich if m >= m1 else Dthin + (Drich - Dthin) * (m - m0) / (m1 - m0); return 1.0 if e <= D else (D / e) ** p
    return f
RULES = [('pair as it is', lambda e, m: 1.0), ('first look only', lambda e, m: 0.0), ('D 6', lin(6)), ('D 4', lin(4)), ('D 3', lin(3)), ('D 2', lin(2)), ('D 4 squared', quad(4)), ('D 3 squared', quad(3)), ('thin 2 rich 8', byev(2, 8)), ('thin 2 rich 6', byev(2, 6)), ('thin 1.5 rich 8', byev(1.5, 8)),
         ('thin 2 rich 8 squared', byev(2, 8, p=2)), ('thin 3 rich 8', byev(3, 8)), ('thin 2 (m<=4) rich 8 (m>=10)', byev(2, 8, 4, 10))]
groups = [('scoreboard as is, dev', lambda r: r['set'] == 'sb' and r['split'] == 'dev'), ('scoreboard as is, unseen', lambda r: r['set'] == 'sb' and r['split'] == 'test'), ('narrow', lambda r: r['set'] == 'sb_n'), ('mono', lambda r: r['set'] == 'sb_m'), ('cut off', lambda r: r['set'] == 'sb_c'), ('excerpt', lambda r: r['set'] == 'sb_e'),
          ('training: rooms', lambda r: r['set'] == 'cases'), ('training: wide and narrow', lambda r: r['set'] == 'cases_w'), ('training: scenes', lambda r: r['set'] == 'cases_s'), ('training: other effects (3 forms)', lambda r: r['set'].startswith('cases_x'))]
wet = [r for r in R if not r['dry'] and r['line'][0] is not None]; dry = [r for r in R if r['dry']]
print('%-30s' % 'rule' + ''.join(' | %-22s' % g[0][:22] for g in groups) + ' | ALL with reverb (n=%d): median mean worse under-30 | no reverb (n=%d): taken away 90th, worst | rule acts on' % (len(wet), len(dry)))
for name, rule in RULES:
    s = '%-30s' % name
    for gl, sel in groups:
        v = np.array([at(r, rule(r['excess'], r['mid'])) for r in wet if sel(r)]); s += ' | %5.1f %5.1f %2.0f%% %3d   ' % (np.median(v), v.mean(), 100 * (v < 0).mean(), (v < -30).sum())
    v = np.array([at(r, rule(r['excess'], r['mid'])) for r in wet]); d = np.array([at(r, rule(r['excess'], r['mid'])) for r in dry]); act = sum(rule(r['excess'], r['mid']) < 1.0 for r in wet); actd = sum(rule(r['excess'], r['mid']) < 1.0 for r in dry)
    s += ' | %5.1f %5.1f %4.1f%% %3d | %5.1f %5.1f | %d + %d' % (np.median(v), v.mean(), 100 * (v < 0).mean(), (v < -30).sum(), np.percentile(d, 90), d.max(), act, actd); print(s)
