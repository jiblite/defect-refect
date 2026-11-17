# The rule "is this lag an echo", fifth round: round 4's data (mkdata4.py: what the app's own engine measures on clips and whole
# sounds, clean, with a synthetic echo, and under every other effect family) plus echoes IN TIME with the sound (mkdata5.py: a
# delay on rhythmic material is nearly always set to the tempo, so its echoes fall on the grid of the sound's own repeats).
# Sources are split as in round 4: the fit uses the training clips and the corpus sounds marked "dev"; the corpus sounds marked
# "test" (and everything made from them) are only judged. The scoreboard's delay cases are never fitted on.
#   python3 fit5.py            judge rule 3 (the p the tool printed when the data was measured), rule 4 (rule4.pkl, when it is
#                              there) and the new fits on the test sources
#   python3 fit5.py save b     .. and write rule5.pkl (the variant with that letter, fitted on the fit sources only) and X5.npy
# In the app: variant b (round 4 and 5 data with equal weight, depth 3, 300 trees), and pOwn 0.6 in Echo.h.
import json, sys, os, glob, pickle, collections, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, '.'); import echo2
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); nan = float('nan')
LEAVE = ('reverb',)                 # a reverb's discrete reflections are echoes of their own kind: neither right nor wrong
def clean(r): return {k: (nan if v is None else v) for k, v in r.items()}
rows = []
for f in sorted(glob.glob('d4_[0-9]*.jsonl')) + sorted(glob.glob('d5_[0-9]*.jsonl')):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
def split(r): return 'fit' if r['kind'] == 'clip' else ('fit' if meta[r['src']]['split'] == 'dev' else 'test')
def synced(r): return r['var'].startswith('sync')
def label(r, c, q):
    """1 = the echo that was put on, 0 = no echo, None = neither (a multiple of the echo time)."""
    if r['var'] == 'clean': return 0
    ratio = q / r['par']['D'][c]
    if abs(ratio - 1) < 0.004: return 1
    if round(ratio) >= 2 and abs(ratio - round(ratio)) < 0.004: return None
    return 0
X = []; y = []; grp = []; sp = []; syn = []; gid = {}
for i, r in enumerate(rows):
    r['split'] = split(r)
    for c in range(2):
        for k, rec in enumerate(r['m']['ch'][c]['recs']):
            rec = clean(rec); r['m']['ch'][c]['recs'][k] = rec; lab = label(r, c, rec['q']); rec['lab'] = lab; rec['row'] = len(X) if lab is not None and r['kind'] not in LEAVE else -1
            if rec['row'] < 0: continue
            X.append(echo2.feats(rec)); y.append(lab); grp.append(gid.setdefault(r['src'], len(gid))); sp.append(r['split'] == 'fit'); syn.append(synced(r))
X = np.array(X, float); y = np.array(y); grp = np.array(grp); sp = np.array(sp); syn = np.array(syn)
print('%d sounds measured (%d clips, %d whole sounds of which %d with an echo in time, %d effect sounds); %d lags for the fit (%d echoes, share %.3f; %d from sounds with an echo in time), %d lags held back' % (
    len(rows), sum(r['kind'] == 'clip' for r in rows), sum(r['kind'] == 'dry' for r in rows), sum(synced(r) for r in rows), sum(r['kind'] not in ('clip', 'dry') for r in rows), sp.sum(), y[sp].sum(), y[sp].mean(), (sp & syn).sum(), (~sp).sum()))
def make(depth=3, iters=150, lr=0.06, leaf=12): return HistGradientBoostingClassifier(max_depth=depth, max_iter=iters, learning_rate=lr, l2_regularization=1.0, min_samples_leaf=leaf)

K = lambda prior: 0.10 / 0.90 / (prior / (1 - prior))
def sure(p, prior): k = K(prior); return p * k / (p * k + 1 - p)
def verdict(r, P, pFound=0.5, pOwn=0.8, pJoin=0.2):
    """The app's policy (Echo.h analyse) with the lags' p from P(rec): -> (p of the surer channel or 0, [lag found per channel or None])"""
    best = [None, None]
    for c in range(2):
        for rec in r['m']['ch'][c]['recs']:
            p = P(rec)
            if best[c] is None or p > best[c][0]: best[c] = (p, rec)
    if best[0] is None and best[1] is None: return 0.0, [None, None]
    lead = 1 if best[0] is None else 0 if best[1] is None else (1 if best[1][0] > best[0][0] else 0)
    if best[lead][0] < pFound: return best[lead][0], [None, None]
    q0 = best[lead][1]['q']; use = [None, None]; use[lead] = best[lead][1]; c = 1 - lead; top = -1
    for rec in r['m']['ch'][c]['recs']:
        p = P(rec); ratio = rec['q'] / q0 if rec['q'] >= q0 else q0 / rec['q']; fits = abs(ratio - round(ratio)) < 0.01
        if p >= (pJoin if fits else pOwn) and p > top: use[c] = rec; top = p
    return best[lead][0], use
def judge(P, prior, which, title, cut=0.65, names=True, pOwn=0.8):
    R = [r for r in rows if r['split'] == which]; res = {}
    def shown(r): p, use = verdict(r, P, pOwn=pOwn); return (p >= 0.5 and sure(p, prior) > cut), p, use
    out = ['%s (%s sources, slider %.0f %%)' % (title, which, cut * 100)]
    for kind, lab in (('clip', 'clean clips'), ('dry', 'clean whole')):
        S = [r for r in R if r['kind'] == kind and r['var'] == 'clean']
        if not S: continue
        n = 0; hot = []
        for r in S:
            s, p, use = shown(r); n += s
            if s: hot.append('%s %.0f ms p %.2f' % ('rec' if kind == 'dry' and meta[r['src']]['kind'] == 'real-recording' else r['name'] if kind == 'dry' else r['name'][3:], max(u['q'] for u in use if u) / 48.0, p))
        out.append('  %-12s %d/%d%s' % (lab, n, len(S), (' (' + ' | '.join(hot[:6]) + ')') if names and hot else '')); res[lab] = (n, len(S))
    for sel, lab in ((lambda r: r['kind'] == 'clip' and r['var'] != 'clean', 'clips+echo'), (lambda r: r['kind'] == 'dry' and r['var'] != 'clean' and not synced(r), 'whole+echo'), (lambda r: r['kind'] == 'dry' and synced(r), 'whole+echo in time')):
        S = [r for r in R if sel(r)]
        if not S: continue
        right = wrong = both = two = 0; byg = collections.defaultdict(lambda: [0, 0])
        for r in S:
            s, p, use = shown(r); ok = bad = False; nok = 0
            if s:
                for c in range(2):
                    if use[c] is None: continue
                    if abs(use[c]['q'] / r['par']['D'][c] - 1) < 0.004: ok = True; nok += 1
                    else: bad = True
            g = abs(r['par']['g']); b = '<0.3' if g < 0.3 else '0.3-0.5' if g < 0.5 else '>=0.5'; byg[b][1] += 1
            if r['par']['D'][0] != r['par']['D'][1]: two += 1; both += (nok == 2 and not bad)
            if ok and not bad: right += 1; byg[b][0] += 1
            elif s: wrong += 1
        out.append('  %-18s n %4d found %5.1f %% wrong time %4.1f %% | by echo level: %s | two echo times: both found %d/%d' % (lab, len(S), 100.0 * right / len(S), 100.0 * wrong / len(S),
                   ', '.join('%s %.0f %%' % (b, 100.0 * byg[b][0] / max(byg[b][1], 1)) for b in ('<0.3', '0.3-0.5', '>=0.5')), both, two)); res[lab] = (right, wrong, len(S))
    fams = sorted({r['kind'] for r in R if r['kind'] not in ('clip', 'dry')}); tot = [0, 0]; line = []
    for fam in fams:
        S = [r for r in R if r['kind'] == fam]; n = 0; hot = []
        for r in S:
            s, p, use = shown(r); n += s
            if s: hot.append('%.0f ms' % (max(u['q'] for u in use if u) / 48.0))
        if fam not in LEAVE: tot[0] += n; tot[1] += len(S)
        if n: line.append('%s %d/%d%s' % (fam, n, len(S), (' (' + ', '.join(hot[:4]) + ')') if hot and names else ''))
    if fams: out.append('  other effects %d/%d (reverb not counted) | %s' % (tot[0], tot[1], ' | '.join(line))); res['others'] = tuple(tot)
    print('\n' + '\n'.join(out)); return res
DLY = [json.loads(l) for l in open('d4_delay.jsonl')]
for r in DLY:
    for c in range(2): r['m']['ch'][c]['recs'] = [clean(x) for x in r['m']['ch'][c]['recs']]
def delaySet(P, prior, title, pOwn=0.8, cut=0.65):
    n = 0; out = []
    for r in DLY:
        p, use = verdict(r, P, pOwn=pOwn); s = p >= 0.5 and sure(p, prior) > cut; n += s
        out.append('%s %s' % (r['name'].replace('dly_', ''), ('/'.join('%.0f' % (u['q'] / 48.0) if u else '-' for u in use) + ' p %.2f' % p) if s else 'NO (p %.2f)' % p))
    print('  %s: delay scoreboard shown at %.0f %% (pOwn %.2f): %d of %d | %s' % (title, cut * 100, pOwn, n, len(DLY), ' | '.join(out))); return n
def batchP(m, extra=()):
    """p for every record of rows + DLY in one call"""
    recs = [rec for r in list(rows) + list(DLY) for c in range(2) for rec in r['m']['ch'][c]['recs']]
    p = m.predict_proba(np.array([echo2.feats(rec) for rec in recs], float))[:, 1]; return {id(rec): float(v) for rec, v in zip(recs, p)}

if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    judge(lambda rec: rec['p'], 0.334, 'test', 'RULE 3 (in the app now)'); delaySet(lambda rec: rec['p'], 0.334, 'rule 3')
    judge(lambda rec: rec['p'], 0.334, 'fit', 'RULE 3 (in the app now)', names=False)
    if os.path.exists('rule4.pkl'):         # written by fit4.py save; not kept in the repo
        m4 = pickle.load(open('rule4.pkl', 'rb')); p4 = batchP(m4)
        judge(lambda rec: p4[id(rec)], 0.3407, 'test', 'RULE 4 (round 4 data only; depth 3, 150 trees)'); delaySet(lambda rec: p4[id(rec)], 0.3407, 'rule 4')
    d4only = sp & ~syn
    variants = [('a: d4+d5, depth 3, 150 trees', {}, 1.0), ('b: d4+d5, depth 3, 300 trees', dict(iters=300), 1.0), ('c: d4+d5 x3, depth 3, 300 trees', dict(iters=300), 3.0),
                ('d: d4+d5 x3, depth 4, 200 trees', dict(depth=4, iters=200), 3.0), ('e: d4+d5 x3, depth 3, 150 trees', {}, 3.0)]
    for name, kw, w in variants:
        if 'quick' in args and name[0] not in 'ac': continue
        sw = np.where(syn, w, 1.0); prior = float(np.average(y[sp], weights=sw[sp]))
        m = make(**kw).fit(X[sp], y[sp], sample_weight=None if w == 1.0 else sw[sp]); pa = batchP(m)
        po = {}; idx = np.where(sp)[0]; pfold = np.zeros(len(X))
        for tr, te in GroupKFold(6).split(X[idx], y[idx], grp[idx]): pfold[idx[te]] = make(**kw).fit(X[idx[tr]], y[idx[tr]], sample_weight=None if w == 1.0 else sw[idx[tr]]).predict_proba(X[idx[te]])[:, 1]
        Ptest = lambda rec: pa[id(rec)]
        Pfit = lambda rec: pfold[rec['row']] if rec.get('row', -1) >= 0 else pa[id(rec)]
        judge(Ptest, prior, 'test', 'RULE 5 %s (prior %.3f)' % (name, prior))
        for po_ in (0.8, 0.6): delaySet(Ptest, prior, 'rule 5 ' + name[0], pOwn=po_)
        judge(Ptest, prior, 'test', 'RULE 5 %s, pOwn 0.6' % name[:1], pOwn=0.6, names=False)
        judge(Pfit, prior, 'fit', 'RULE 5 %s, out of fold' % name[:1], names=False)
        if 'save' in args and args[args.index('save') + 1] == name[0]:
            pickle.dump(m, open('rule5.pkl', 'wb')); np.save('X5.npy', X); print('rule5.pkl written (variant %s, fitted on %d lags of the fit sources; prior %.4f)' % (name, sp.sum(), prior))
