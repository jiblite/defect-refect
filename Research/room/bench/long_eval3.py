# A long sound judged in pieces of 6 s, each looked at as a sound of its own (the app's roomInPieces, printed by rsure.cpp
# with the look "pieces"), against the rule on the whole sound and against the whole sound's evidence taken stretch by
# stretch (long_eval.py, long_eval2.py).
#   python3 long_eval3.py [tag] [--pieces]     (tag: default x45f; --pieces: every sound's pieces)     CUT=0.65 in the environment: where REVERB is shown
# Columns: whole = the rule on the whole sound (the app up to 0.4.4 and on sounds of up to 9 s); smean, smean|send = the
# whole sound's evidence by 6 s stretches: their average, and the larger of the average and the last stretch; pmean = the
# average of the pieces, pend = the last piece, app = the app's verdict (the larger of the two; nothing when the whole
# sound has no room), nogate = the same without asking for a room on the whole sound.
import sys, os, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '/tmp/claude-0/v2/roomsure')
show = '--pieces' in sys.argv; sys.argv = [a for a in sys.argv if a != '--pieces']
if len(sys.argv) < 2: sys.argv.append('x45f')
import long_eval as LE, look2
R = LE.load(); CUT = float(os.environ.get('CUT', '0.65')); NONE = LE.NONE
def cand(r):
    a = r['j']['as']; v = dict(whole=r['v']['whole'], smean=r['v']['mean568']); P = a.get('pieces') or []
    sp6 = a.get('spans568') or []; v['send'] = LE.rule(sp6[-1][0], sp6[-1][1]) if a['found'] and sp6 else NONE; v['smean|send'] = max(v['smean'], v['send'])
    if not P:
        for k in ('pmean', 'pend', 'app', 'nogate'): v[k] = v['whole']
        return v
    s = np.array([p[1] for p in P]); w = np.array([p[-1] for p in P], float); mean = float((s * w).sum() / w.sum()) if w.sum() > 0 else NONE; end = float(s[-1]) if w[-1] >= 0.2 * w.max() else 0.0
    v['pmean'] = mean if a['found'] else NONE; v['pend'] = end if a['found'] else NONE; v['nogate'] = max(mean, end); v['app'] = a['verdict'] if 'verdict' in a else (max(mean, end) if a['found'] else NONE)
    if 'inpieces' in a: assert abs(a['inpieces'] - max(mean, end)) < 2e-4, (r['path'], a['inpieces'], mean, end)          # the app's own arithmetic against this script's
    return v
for r in R: r['c'] = cand(r)
keys = ['whole', 'smean', 'smean|send', 'pmean', 'pend', 'app', 'nogate']
if __name__ == '__main__':
    G = collections.OrderedDict()
    for r in R: G.setdefault((LE.name(r), r['split']), []).append(r)
    print('%d looks, REVERB shown at %.0f %%' % (len(R), 100 * CUT)); print('%-46s %-5s %3s | %s' % ('kind', 'split', 'n', ' '.join('%11s' % k for k in keys)))
    for (k, sp), g in sorted(G.items()): print('%-46s %-5s %3d | %s' % (k, sp, len(g), ' '.join('%10.0f%%' % (100 * np.mean([r['c'][q] > CUT for r in g])) for q in keys)))
    for lab, sel in (('ALL without reverb', lambda r: r['wet'] == 0), ('  clean clips looped', lambda r: r['kind'] == 'loopc'), ('  clips in a row', lambda r: r['kind'] in ('catc', 'catfx')), ('    clean', lambda r: r['kind'] == 'catc'), ('    with sweep, phaser or chorus', lambda r: r['kind'] == 'catfx'),
                     ('ALL with reverb', lambda r: r['wet'] == 1), ('  cut off', lambda r: r['wet'] == 1 and int(r['cut']) > 0), ('  rings out', lambda r: r['wet'] == 1 and not int(r['cut']))):
        for sp in (None, 'dev', 'test'):
            g = [r for r in R if sel(r) and (sp is None or r['split'] == sp)]
            if g and (sp is None or lab.startswith('ALL')): print('%-46s %-5s %3d | %s' % (lab, sp or 'all', len(g), ' '.join('%6d %3.0f%%' % (sum(r['c'][q] > CUT for r in g), 100 * np.mean([r['c'][q] > CUT for r in g])) for q in keys)))
    sb = {r['case']: r for r in look2.load('sb', 'x45') if r['fam'] == 'clean'}; lc = [r for r in R if r['kind'] == 'loopc' and not int(r['cut']) and r['dry'] in sb]
    print('clean clip six times over against the clip itself (n=%d; the clip alone shows REVERB on %d):' % (len(lc), sum(sb[r['dry']]['v_as'] > CUT for r in lc)))
    for q in keys:
        a = np.array([sb[r['dry']]['v_as'] > CUT for r in lc]); b = np.array([r['c'][q] > CUT for r in lc]); print('  %-12s shown %2d | same verdict %2d | only as a loop %d, only as a clip %d' % (q, b.sum(), (a == b).sum(), (b & ~a).sum(), (a & ~b).sum()))
    # what the number says: of the sounds the app calls so sure, how many have a reverb
    print('what the app\'s number says on these sounds: ' + '; '.join('%s: %d of %d have a reverb' % (lab, sum(r['wet'] for r in g), len(g)) for lab, g in ((lab, [r for r in R if lo < r['c']['app'] <= hi]) for lab, lo, hi in (('up to 35 %', -1, .35), ('35 to 65 %', .35, .65), ('65 to 80 %', .65, .8), ('80 to 90 %', .8, .9), ('over 90 %', .9, 2))) if g))
    if show:
        for r in R:
            P = r['j']['as'].get('pieces') or []; m = max([p[-1] for p in P] or [1])
            print('%-34s %-16s cut %-2d wet %d | whole %.2f app %.2f | %s' % (os.path.basename(r['path'])[:34], r['tool'][:16], int(r['cut']), r['wet'], r['c']['whole'], r['c']['app'], ' '.join('%.0f%s' % (100 * p[1], '*' if p[-1] < 0.2 * m else '') for p in P)))
