# The final rule on the stored statistics. The clean corpus sounds and the synthetic flangers were run before the level test
# above 2 * SR / move existed; that one number is taken from levelb.json (levelb.py: computed afterwards from the stored lines,
# for every line that passes the rest of the rule). The clean training clips (stats_train.jsonl) were run with the final engine.
import sys, json, os, collections; sys.argv = sys.argv[:1]; sys.path.insert(0, '/tmp/claude-0/v2/sweep'); os.chdir('/tmp/claude-0/v2/sweep')
import rule, sweep, numpy as np
LB = json.load(open('levelb.json'))
for part, rows in (('clean', rule.clean), ('synth', rule.synth)):
    for r in rows:
        for c, cl in enumerate(r['cands']):
            for k, f in enumerate(cl):
                v = LB.get('%s/%s/%d/%d' % (part, r['id'], c, k))
                if v: f['level_b'] = v['b2']
def R(f): return f.get('g', 0) != 0 and 'level_b' in f and 'static' in f and sweep.accept(f)
def first(f): return f.get('g', 0) != 0 and 'level' in f and rule.make()(f)
for name, rl in (('first rule (strength, movement, line gone, plain level)', first), ('final rule', R)):
    print('==', name)
    for part, rows in (('clean corpus', rule.clean), ('clean training clips', rule.train)):
        for sp in ('dev', 'test', 'train'):
            rr = [r for r in rows if r['split'] == sp]
            if not rr: continue
            fa = [r for r in rr if rule.decide(r['cands'], rl) is not None]; ln = sum(1 for r in rr for cl in r['cands'] for f in cl if rl(f))
            print('   %-22s %-5s falsely reported %d of %d sounds (single lines that pass: %d of %d followed up)  %s' % (part, sp, len(fa), len(rr), ln, sum(1 for r in rr for cl in r['cands'] for f in cl if f.get('g', 0) != 0), ' '.join(r['id'] for r in fa)))
    tot = []
    for sp in ('dev', 'test'):
        rr = [r for r in rule.synth if r['split'] == sp]; got = [(r, rule.decide(r['cands'], rl)) for r in rr]; sc = [float(np.mean([f['out_alone'] for f in ch])) for r, ch in got if ch is not None]; tot += sc
        al = np.array(sc + [0.0] * (len(rr) - len(sc)))
        print('   synthetic flangers     %-5s found %d of %d | where found: out median %.1f mean %.1f min %.1f, worse than nothing %d | over all: median %.1f mean %.1f' % (sp, len(sc), len(rr), np.median(sc), np.mean(sc), np.min(sc), sum(v < 0 for v in sc), np.median(al), al.mean()))
    t = collections.Counter()
    for r in rule.synth:
        for cl in r['cands']:
            for f in cl:
                if 'out_alone' in f: t[(bool(rl(f)), f['out_alone'] > 0)] += 1
    print('   single lines on the synthetic flangers (good: taking it out alone scores above 0): accepted good %d, accepted bad %d, refused good %d, refused bad %d' % (t[(True, True)], t[(True, False)], t[(False, True)], t[(False, False)]))
# who is not found
print('== synthetic flangers the final rule does not find, by what stops them')
why = collections.Counter(); kinds = collections.Counter(); kf = collections.Counter()
for r in rule.synth:
    kinds[r['kind']] += 1
    if rule.decide(r['cands'], R) is not None: kf[r['kind']] += 1; continue
    fl = [f for cl in r['cands'] for f in cl if f.get('g', 0) != 0]; good = [f for f in fl if f.get('d_err', 99) < 2.5]
    if not fl: why['no line strong enough to follow up'] += 1
    elif not good: why['only wrong lines (none within 2.5 samples of the true one)'] += 1
    elif any(R(f) for f in good): why['true line passes on one channel only, or signs differ'] += 1
    else:
        f = max(good, key=lambda f: f.get('out_alone', -999)); p = r['par']
        w = 'weak' if abs(f['g']) < 0.2 else 'jumps' if f['fast'] > sweep.JUMPS else 'stands still' if f['static'] > sweep.STILL[1 if f['periodic'] else 0] else 'level' if 'level_b' in f and not sweep.passes(f['res'], f['level_b'], f['expect']) else 'rings' if abs(f['fb']) > 0.75 else 'other'
        why['true line followed up but refused: ' + w] += 1
for k, v in why.most_common(): print('   %2d  %s' % (v, k))
print('   found by kind of sound: ' + ', '.join('%s %d/%d' % (k, kf[k], kinds[k]) for k in sorted(kinds)))
