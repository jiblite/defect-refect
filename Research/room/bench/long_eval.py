# The REVERB sureness on long sounds (prep_long.py), by the rule as it is (evidence counted over the whole sound) and
# stretch by stretch (the fitted rule on every stretch of the sound, then one verdict from the stretches).
#   python3 long_eval.py [tag]      (tag: the run of the study tool over long_list.txt with the look "pieces", ./run4.sh long Resources x45f 3 pieces; default x45f)
import sys, os, json, collections, numpy as np
sys.path.insert(0, '/tmp/claude-0/v2/roomsure'); import look2
D = '/tmp/claude-0/v2/roomsure/'; TAG = sys.argv[1] if len(sys.argv) > 1 else 'x45f'; CUT = 0.65
A0, AD, AB, NONE = 1.902, 0.847, 2.649, 0.22
def rule(deep, bands): return NONE if bands < 1 else 1.0 / (1.0 + np.exp(-(A0 + AD * np.log10(deep + 0.003) + AB * bands / 41.0)))
def wq(v, w, q):
    """Weighted quantile: in rising order, the first value at which the weight so far reaches q of all."""
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w, float)[o]; c = np.cumsum(w) / max(w.sum(), 1e-30); return float(v[min(np.searchsorted(c, q - 1e-12), len(v) - 1)])
def verdicts(a):
    out = dict(whole=a['sure'] if a['found'] else NONE)
    for span in (852, 568, 426):
        sp = a.get('spans%d' % span) or []
        if not a['found'] or not sp:
            for k in ('med', 'q60', 'q70', 'mean', 'logit'): out['%s%d' % (k, span)] = NONE
            continue
        s = np.array([rule(d, b) for d, b, l in sp]); w = np.array([l for d, b, l in sp], float)
        out['med%d' % span] = wq(s, w, 0.5); out['q60%d' % span] = wq(s, w, 0.6); out['q70%d' % span] = wq(s, w, 0.7); out['mean%d' % span] = float((s * w).sum() / max(w.sum(), 1e-30))
        lg = np.log(s / (1 - s)); out['logit%d' % span] = float(1 / (1 + np.exp(-(lg * w).sum() / max(w.sum(), 1e-30))))
    return out
def load():
    rows = json.load(open(D + 'long_rows.json')); out = {}
    for l in open(D + 'long_out_%s.jsonl' % TAG):
        j = json.loads(l); out[(os.path.basename(j['path']), int(j.get('cut', 0)))] = j
    R = []
    for r in rows:
        j = out.get((os.path.basename(r['path']), int(r['cut'])))
        if j is not None: R.append(dict(r, j=j, v=verdicts(j['as'])))
    return R
def name(r):
    c = int(r['cut'])
    if r['kind'] == 'loopc': return 'clean clip six times over (36 s)' if not c else 'clean clip three times over (18 s)'
    if r['kind'] == 'loopr': return 'reverb on a looped clip, rings out (39 s)' if not c else 'reverb on a looped clip, cut off at %d s' % c
    if r['kind'] == 'catc': return 'clips in a row, clean (30 to 48 s)'
    if r['kind'] == 'catfx': return 'clips in a row with sweep, phaser or chorus'
    return 'reverb on clips in a row, rings out' if not c else 'reverb on clips in a row, cut off'
if __name__ == '__main__':
    R = load(); keys = ['whole'] + ['%s%d' % (k, s) for s in (852, 568, 426) for k in ('med', 'q60', 'q70', 'mean', 'logit')]
    print('%d looks. Shown at %.0f %% by each way of judging (whole = as the rule is; med/q60/q70 = middle, 60th, 70th hundredth of the stretches by weight; mean, logit = averages; 852/568/426 = frames per stretch)' % (len(R), 100 * CUT))
    G = collections.OrderedDict()
    for r in R: G.setdefault((name(r), r['split']), []).append(r)
    print('%-46s %-5s %3s | %s' % ('kind', 'split', 'n', ' '.join('%8s' % k for k in keys)))
    for (k, sp), g in sorted(G.items()):
        print('%-46s %-5s %3d | %s' % (k, sp, len(g), ' '.join('%7.0f%%' % (100 * np.mean([r['v'][q] > CUT for r in g])) for q in keys)))
    for lab, sel in (('ALL without reverb', lambda r: r['wet'] == 0), ('ALL with reverb', lambda r: r['wet'] == 1), ('with reverb, cut off', lambda r: r['wet'] == 1 and int(r['cut']) > 0), ('with reverb, rings out', lambda r: r['wet'] == 1 and not int(r['cut']))):
        for sp in ('dev', 'test', 'mixed', None):
            g = [r for r in R if sel(r) and (sp is None or r['split'] == sp)]
            if g: print('%-46s %-5s %3d | %s' % (lab, sp or 'all', len(g), ' '.join('%7.0f%%' % (100 * np.mean([r['v'][q] > CUT for r in g])) for q in keys)))
    # a clean clip played six times over against the clip itself (the 6 s look of the x45 run)
    sb = {r['case']: r for r in look2.load('sb', 'x45') if r['fam'] == 'clean'}
    lc = [r for r in R if r['kind'] == 'loopc' and not int(r['cut']) and r['dry'] in sb]
    print('clean clip six times over against the clip itself (n=%d): the clip alone shows REVERB on %d' % (len(lc), sum(sb[r['dry']]['v_as'] > CUT for r in lc)))
    for q in keys:
        a = np.array([sb[r['dry']]['v_as'] > CUT for r in lc]); b = np.array([r['v'][q] > CUT for r in lc]); d = np.array([abs(sb[r['dry']]['v_as'] - r['v'][q]) for r in lc])
        print('  %-9s shown %2d | same verdict %2d of %d | shown only as a loop %d, only as a clip %d | sureness apart: median %.2f, largest %.2f' % (q, b.sum(), (a == b).sum(), len(lc), (b & ~a).sum(), (a & ~b).sum(), np.median(d), d.max()))
