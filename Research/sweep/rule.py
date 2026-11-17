# The rule judged on the stored numbers (stats_clean.jsonl, stats_synth.jsonl, stats_train.jsonl, with the lines in lines_*.pkl):
# how many clean sounds are falsely reported, how many synthetic flangers are found and what taking them out scores, and how
# the rule's verdict on single lines compares with what taking the line out really does (out_alone > 0: good, else bad).
#   python3 rule.py [name=value ...]        thresholds to try (see RULE below); without arguments: the rule in sweep.py
import sys, json, os, pickle, numpy as np, collections
os.chdir('/tmp/claude-0/v2/sweep'); sys.path.insert(0, '/tmp/claude-0/v2/sweep')
import sweep

RULE = dict(gmin=0.2, move=12.0, fbmax=0.75, c1max=1.0, res=0.3, seen=0.15, reach=0.0, qlo=0.35, qhi=2.5, tilt=0, jumps=1.0, still=1.0, jumps_p=-1.0, still_p=-1.0, conc=1.0, hl=0.0, jumps2=1.0, both=0)
def make(**kw):
    p = dict(RULE); p.update(kw)
    def rule(f):
        if f.get('g', 0) == 0 or 'level' not in f: return False
        if abs(f['g']) < p['gmin'] or f['move'] < p['move'] or abs(f['fb']) > p['fbmax'] or abs(f['c1_first']) > p['c1max'] or abs(f['res']) > p['res'] or f['seen'] < p['seen']: return False
        if f['dmax'] < p['reach'] or f['expect'] > -0.15: return False
        q = (f['level_b'] if p.get('band') else f['level_t'] if p['tilt'] else f['level']) / f['expect']
        if not (p['qlo'] <= q <= p['qhi']): return False
        if p['both'] and not (p['qlo'] <= f['level'] / f['expect'] <= p['qhi']): return False
        per = f.get('periodic', 0) > 0
        if f.get('fast', 0) > (p['jumps_p'] if per and p['jumps_p'] >= 0 else p['jumps']) or f.get('static', 0) > (p['still_p'] if per and p['still_p'] >= 0 else p['still']): return False
        if f.get('jumps2', 0) > p['jumps2']: return False
        if f.get('conc', 0) > p['conc']: return False
        if p['hl'] > 0 and f.get('lo', 0) > 0.2 and f.get('hi', 0) < p['hl'] * f['lo']: return False
        return True
    return rule

def decide(cands, rule):
    """find()'s decision from the stored candidates of a sound: the chosen candidate per channel, or None."""
    ch = []
    for cl in cands:
        good = [f for f in cl if rule(f)]
        ch.append(max(good, key=lambda f: abs(f['g']) * f['seen']) if good else None)
    if not ch or any(c is None for c in ch): return None
    if len(ch) == 2 and ch[0]['g'] * ch[1]['g'] < 0: return None
    return ch

def shape(l):
    d, w = l['d'].astype(float), l['w'].astype(float); s = np.abs(np.diff(d)); m = (w[1:] > 0.05) & (w[:-1] > 0.05)
    path = l['path'].astype(float); k1 = np.abs(np.diff(path)); k2 = np.abs(np.diff(path) - np.diff(d)); tv = k1.sum()
    r = dict(jumps2=float(k1[k2 >= 4].sum() / tv) if tv > 0 else 0.0, jumps1=float(k1[k1 >= 4].sum() / tv) if tv > 0 else 0.0)
    if m.sum() < 8 or s[m].sum() <= 0: r.update(conc=1.0, quiet=1.0); return r
    v = np.sort(s[m])[::-1]; k = max(1, int(round(0.2 * len(v))))
    r.update(conc=float(v[:k].sum() / v.sum()), quiet=float((s[m] < 0.05).mean())); return r
def load(part):
    p = 'stats_%s.jsonl' % part
    if not os.path.exists(p): return []
    rows = [json.loads(l) for l in open(p)]; lp = 'lines_%s.pkl' % part
    if os.path.exists(lp):
        try: L = pickle.load(open(lp, 'rb'))
        except Exception: L = {}
        for r in rows:
            for c, cl in enumerate(r['cands']):
                for k, f in enumerate(cl):
                    l = L.get(r['id'], [[]] * 2)[c][k] if r['id'] in L and c < len(L[r['id']]) and k < len(L[r['id']][c]) else None
                    if l is not None: f.update(shape(l))
    return rows
clean, synth, train = load('clean'), load('synth'), load('train')

def report(name, rule, show=True, quiet=False):
    res = {}
    if not quiet: print('--', name)
    for part, rows in (('clean corpus', clean), ('clean training clips', train)):
        for sp in ('dev', 'test', 'train'):
            rr = [r for r in rows if r['split'] == sp]
            if not rr: continue
            fa = [r for r in rr if decide(r['cands'], rule) is not None]; res['fa_' + sp] = (len(fa), len(rr))
            if not quiet: print('   %-22s %-5s falsely reported %d of %d%s' % (part, sp, len(fa), len(rr), ('   ' + ' '.join(r['id'] for r in fa)) if show else ''))
    for sp in ('dev', 'test'):
        rr = [r for r in synth if r['split'] == sp]
        if not rr: continue
        got = [(r, decide(r['cands'], rule)) for r in rr]; sc = [float(np.mean([f.get('out_alone', 0.0) for f in ch])) for r, ch in got if ch is not None]
        allsc = np.array(sc + [0.0] * (len(rr) - len(sc))); res['found_' + sp] = (len(sc), len(rr)); res['worse_' + sp] = sum(s < 0 for s in sc)
        if not quiet: print('   synthetic flangers     %-5s found %d of %d | where found: out median %.1f mean %.1f, worse than nothing %d | over all: median %.1f mean %.1f' % (
            sp, len(sc), len(rr), np.median(sc) if sc else 0, np.mean(sc) if sc else 0, sum(s < 0 for s in sc), np.median(allsc), allsc.mean()))
    # single lines: the rule's verdict against what taking the line out does
    t = collections.Counter()
    for r in synth:
        for cl in r['cands']:
            for f in cl:
                if 'out_alone' in f: t[(bool(rule(f)), f['out_alone'] > 0)] += 1
    res['lines'] = t
    if not quiet and t: print('   single lines on the synthetic flangers: accepted and good %d, accepted and bad %d, refused and good %d, refused and bad %d' % (t[(True, True)], t[(True, False)], t[(False, True)], t[(False, False)]))
    return res

if __name__ == '__main__':
    kw = {a.split('=')[0]: float(a.split('=')[1]) for a in sys.argv[1:]}
    if kw: report('rule %s' % kw, make(**kw))
    else: report('the rule in sweep.py', lambda f: f.get('g', 0) != 0 and 'level_b' in f and sweep.accept(f))
