# The rule "is this lag an echo", fourth round: fitted on what the app's own engine measures (drecho measure, mkdata4.py) on
# 6 s clips AND on whole sounds: clean ones, ones with a synthetic echo, and the wet sounds of every other effect family (where a
# reported echo is a false report). Sources are split: the fit uses the training clips and the corpus sounds marked "dev"; the
# corpus sounds marked "test" (and everything made from them) are only judged. The scoreboard's delay cases are never fitted on.
#   python3 fit4.py            judge rule 3 (the p the tool printed) and the new fit on the test sources
#   python3 fit4.py save       .. and write rule4.pkl (fitted on the fit sources only) and X4.npy (for export_rule.py's check)
import json, sys, glob, pickle, collections, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, '.'); import echo2
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); nan = float('nan')
LEAVE = ('reverb',)                 # a reverb's discrete reflections are echoes of their own kind: neither right nor wrong
def clean(r): return {k: (nan if v is None else v) for k, v in r.items()}
rows = []
for f in sorted(glob.glob('d4_[0-9]*.jsonl')):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
def combish(r):                     # a tone case whose "tone" is a comb of 30 ms or more would be a true echo; none is made that long
    return False
def split(r): return 'fit' if r['kind'] == 'clip' else ('fit' if meta[r['src']]['split'] == 'dev' else 'test')
def label(r, c, q):
    """1 = the echo that was put on, 0 = no echo, None = neither (a multiple of the echo time)."""
    if r['var'] == 'clean': return 0
    ratio = q / r['par']['D'][c]
    if abs(ratio - 1) < 0.004: return 1
    if round(ratio) >= 2 and abs(ratio - round(ratio)) < 0.004: return None
    return 0
X = []; y = []; grp = []; own = []; sp = []; gid = {}
for i, r in enumerate(rows):
    r['split'] = split(r)
    for c in range(2):
        for k, rec in enumerate(r['m']['ch'][c]['recs']):
            rec = clean(rec); r['m']['ch'][c]['recs'][k] = rec; lab = label(r, c, rec['q']); rec['lab'] = lab; rec['row'] = len(X) if lab is not None and r['kind'] not in LEAVE else -1
            if rec['row'] < 0: continue
            X.append(echo2.feats(rec)); y.append(lab); grp.append(gid.setdefault(r['src'], len(gid))); own.append(i); sp.append(r['split'] == 'fit')
X = np.array(X, float); y = np.array(y); grp = np.array(grp); sp = np.array(sp)
print('%d sounds measured (%d clips, %d whole sounds, %d effect sounds); %d lags for the fit (%d echoes, share %.3f), %d lags held back for judging' % (
    len(rows), sum(r['kind'] == 'clip' for r in rows), sum(r['kind'] == 'dry' for r in rows), sum(r['kind'] not in ('clip', 'dry') for r in rows), sp.sum(), y[sp].sum(), y[sp].mean(), (~sp).sum()))
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
def judge(P, prior, which, title, cuts=(0.5, 0.65, 0.8), names=True):
    print('\n%s (%s sources)' % (title, which))
    R = [r for r in rows if r['split'] == which]
    def shown(r, cut): p, use = verdict(r, P); return (p >= 0.5 and sure(p, prior) > cut), p, use
    # clean sounds
    for kind, lab in (('clip', 'clean 6 s clips'), ('dry', 'clean whole sounds')):
        S = [r for r in R if r['kind'] == kind and r['var'] == 'clean']
        if not S: continue
        row = '  %-28s n %4d | shown at slider' % (lab, len(S)); hot = []
        for cut in cuts:
            n = 0
            for r in S:
                s, p, use = shown(r, cut); n += s
                if s and cut == 0.65: hot.append('%s %.0f ms p %.2f' % ('rec' if kind == 'dry' and meta[r['src']]['kind'] == 'real-recording' else r['name'] if kind == 'dry' else r['name'][3:], max(u['q'] for u in use if u) / 48.0, p))
            row += ' %.0f %%: %3d (%4.1f %%)' % (cut * 100, n, 100.0 * n / len(S))
        print(row + ('   ' + ' | '.join(hot[:8]) if names and hot else ''))
    # sounds with an echo put on
    for kind, lab in (('clip', '6 s clips with an echo'), ('dry', 'whole sounds with an echo')):
        S = [r for r in R if r['kind'] == kind and r['var'] != 'clean']
        if not S: continue
        for cut in cuts[1:2]:
            right = wrong = 0; byg = collections.defaultdict(lambda: [0, 0])
            for r in S:
                s, p, use = shown(r, cut); ok = bad = False
                if s:
                    for c in range(2):
                        if use[c] is None: continue
                        if abs(use[c]['q'] / r['par']['D'][c] - 1) < 0.004: ok = True
                        else: bad = True
                g = abs(r['par']['g']); b = '<0.3' if g < 0.3 else '0.3-0.5' if g < 0.5 else '>=0.5'; byg[b][1] += 1
                if ok and not bad: right += 1; byg[b][0] += 1
                elif s: wrong += 1
            print('  %-28s n %4d | at slider %.0f %%: found %5.1f %%, with a wrong echo time on a channel %4.1f %% | by loudness of the echo: %s' % (lab, len(S), cut * 100, 100.0 * right / len(S), 100.0 * wrong / len(S),
                  ', '.join('%s %.0f %%' % (b, 100.0 * byg[b][0] / max(byg[b][1], 1)) for b in ('<0.3', '0.3-0.5', '>=0.5'))))
    # the other effect families
    fams = sorted({r['kind'] for r in R if r['kind'] not in ('clip', 'dry')}); tot = [0, 0]; line = []
    for fam in fams:
        S = [r for r in R if r['kind'] == fam]; n = 0; hot = []
        for r in S:
            s, p, use = shown(r, 0.65); n += s
            if s: hot.append('%.0f ms' % (max(u['q'] for u in use if u) / 48.0))
        if fam not in LEAVE: tot[0] += n; tot[1] += len(S)
        line.append('%s %d/%d%s' % (fam, n, len(S), (' (' + ', '.join(hot[:4]) + ')') if hot and names else ''))
    if fams: print('  other effects, shown at 65 %%: %d of %d (reverb not counted) | %s' % (tot[0], tot[1], ' | '.join(line)))
def delaySet(P, prior, title):
    D = [json.loads(l) for l in open('d4_delay.jsonl')] if glob.glob('d4_delay.jsonl') else []
    if not D: return
    n = 0; out = []
    for r in D:
        for c in range(2): r['m']['ch'][c]['recs'] = [clean(x) for x in r['m']['ch'][c]['recs']]
        p, use = verdict(r, P); s = p >= 0.5 and sure(p, prior) > 0.65; n += s; out.append('%s %s' % (r['name'].replace('dly_', ''), ('%.0f ms p %.2f' % (max(u['q'] for u in use if u) / 48.0, p)) if s else 'no (p %.2f)' % p))
    print('  %s: the scoreboard\'s delay sounds shown at 65 %%: %d of %d | %s' % (title, n, len(D), ' | '.join(out)))

if __name__ == '__main__':
    # rule 3 as the tool has it (its p is in the records; its prior 0.334)
    judge(lambda rec: rec['p'], 0.334, 'test', 'RULE 3 (in the app now)'); delaySet(lambda rec: rec['p'], 0.334, 'rule 3')
    judge(lambda rec: rec['p'], 0.334, 'fit', 'RULE 3 (in the app now)', names=False)
    prior = float(y[sp].mean())
    for name, kw in (('depth 3, 150 trees', {}), ('depth 3, 300 trees', dict(iters=300)), ('depth 4, 200 trees', dict(depth=4, iters=200)), ('depth 2, 300 trees', dict(depth=2, iters=300))):
        if 'quick' in sys.argv and kw: continue
        m = make(**kw).fit(X[sp], y[sp]); p = m.predict_proba(X)[:, 1]
        # out of fold on the fit sources (grouped by source sound), so that they can be judged too
        po = p.copy(); idx = np.where(sp)[0]
        for tr, te in GroupKFold(6).split(X[idx], y[idx], grp[idx]): po[idx[te]] = make(**kw).fit(X[idx[tr]], y[idx[tr]]).predict_proba(X[idx[te]])[:, 1]
        P = lambda rec: po[rec['row']] if rec['row'] >= 0 else float(m.predict_proba(np.array([echo2.feats(rec)], float))[0, 1])
        judge(P, prior, 'test', 'RULE 4, %s (prior %.3f)' % (name, prior)); delaySet(lambda rec: float(m.predict_proba(np.array([echo2.feats(rec)], float))[0, 1]), prior, 'rule 4')
        judge(P, prior, 'fit', 'RULE 4, %s, out of fold' % name, names=False)
        if 'save' in sys.argv and not kw:
            pickle.dump(m, open('rule4.pkl', 'wb')); np.save('X4.npy', X); print('rule4.pkl written (fitted on %d lags of the fit sources; prior %.4f)' % (sp.sum(), prior))
