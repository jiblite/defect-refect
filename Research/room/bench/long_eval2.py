# Candidates for one verdict on a long sound from its stretches (see long_eval.py): the average over 6 s stretches, the
# look at the file's last 9 s as a sound of its own, and the larger of the two.
import sys, os, collections, numpy as np
sys.path.insert(0, '/tmp/claude-0/v2/roomsure'); import long_eval as LE, look2
R = LE.load(); CUT = float(os.environ.get('CUT', '0.65'))
def cand(r):
    a = r['j']['as']; v = dict(whole=r['v']['whole'], mean568=r['v']['mean568'], med568=r['v']['med568'], logit568=r['v']['logit568'], mean852=r['v']['mean852'])
    sp = a.get('spans852') or []; v['end852'] = LE.rule(sp[-1][0], sp[-1][1]) if a['found'] and sp else LE.NONE
    sp6 = a.get('spans568') or []; v['end568'] = LE.rule(sp6[-1][0], sp6[-1][1]) if a['found'] and sp6 else LE.NONE
    v['mean568|end852'] = max(v['mean568'], v['end852']); v['med568|end852'] = max(v['med568'], v['end852']); v['logit568|end852'] = max(v['logit568'], v['end852']); v['mean852|end852'] = max(v['mean852'], v['end852'])
    return v
for r in R: r['c'] = cand(r)
keys = ['whole', 'mean568', 'med568', 'end852', 'mean568|end852', 'med568|end852', 'logit568|end852', 'mean852|end852']
G = collections.OrderedDict()
for r in R: G.setdefault((LE.name(r), r['split']), []).append(r)
print('shown at %.0f %%' % (100 * CUT)); print('%-46s %-5s %3s | %s' % ('kind', 'split', 'n', ' '.join('%15s' % k for k in keys)))
for (k, sp), g in sorted(G.items()): print('%-46s %-5s %3d | %s' % (k, sp, len(g), ' '.join('%14.0f%%' % (100 * np.mean([r['c'][q] > CUT for r in g])) for q in keys)))
for lab, sel in (('ALL without reverb', lambda r: r['wet'] == 0), ('  loops', lambda r: r['kind'] == 'loopc'), ('  clips in a row', lambda r: r['kind'] in ('catc', 'catfx')), ('ALL with reverb', lambda r: r['wet'] == 1), ('  cut off', lambda r: r['wet'] == 1 and int(r['cut']) > 0), ('  rings out', lambda r: r['wet'] == 1 and not int(r['cut']))):
    g = [r for r in R if sel(r)]; print('%-46s %-5s %3d | %s' % (lab, 'all', len(g), ' '.join('%10d %3.0f%%' % (sum(r['c'][q] > CUT for r in g), 100 * np.mean([r['c'][q] > CUT for r in g])) for q in keys)))
sb = {r['case']: r for r in look2.load('sb', 'x45') if r['fam'] == 'clean'}; lc = [r for r in R if r['kind'] == 'loopc' and not int(r['cut']) and r['dry'] in sb]
print('clean clip six times over against the clip itself (n=%d; the clip alone shows REVERB on %d):' % (len(lc), sum(sb[r['dry']]['v_as'] > CUT for r in lc)))
for q in keys:
    a = np.array([sb[r['dry']]['v_as'] > CUT for r in lc]); b = np.array([r['c'][q] > CUT for r in lc]); print('  %-16s shown %2d | same verdict %2d | only as a loop %d, only as a clip %d' % (q, b.sum(), (a == b).sum(), (b & ~a).sum(), (a & ~b).sum()))
