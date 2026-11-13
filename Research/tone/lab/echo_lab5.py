# Rule search for the echo finder with three-part and six-part steadiness.
import json, numpy as np
z = json.load(open('work/echo3_clean.json')); files = z['files']; R = z['R']; kind = np.array([f.split('_')[1] for f in files])
def feats(c):
    d, g, g2, agree, med = c[:5]; p3 = np.array(c[5:8]); p6 = np.array(c[8:14]); s3 = np.sign(p3) == np.sign(g); s6 = np.sign(p6) == np.sign(g)
    return dict(d=d, g=g, r2=g2 / (g * g + 1e-9), agree=agree, pmin=(np.abs(p3).min() / abs(g)) if s3.all() else 0.0, n6=int((s6 & (np.abs(p6) >= 0.4 * abs(g))).sum()), n6s=int(s6.sum()))
clean = [[feats(c) for c in r[0]] for r in R]; sims = [[(d0, g0, [feats(c) for c in cands]) for d0, g0, cands in r[1:]] for r in R]
print('simulated reflections among the candidates: %.0f%%' % (100 * np.mean([any(abs(c['d'] - d0) < 0.6 for c in cs) for s in sims for d0, g0, cs in s])))
def rates(rule):
    fa = np.array([any(rule(c) for c in cs) for cs in clean]); out = [100 * fa.mean()] + [100 * fa[kind == k].mean() for k in ('real-recording', 'fluid-melody', 'sample-loop', 'synthetic')]
    for j in range(3): out.append(100 * np.mean([any(abs(c['d'] - s[j][0]) < 0.6 and rule(c) for c in s[j][2]) for s in sims]))
    wrong = 100 * np.mean([any(abs(c['d'] - s[j][0]) >= 0.6 and rule(c) and abs(c['g']) > max([abs(x['g']) for x in s[j][2] if abs(x['d'] - s[j][0]) < 0.6 and rule(x)] + [0]) for c in s[j][2]) for s in sims for j in range(3)]); out.append(wrong)
    return out
print('%-58s false alarms: all real melody loops synth | found g=0.30 0.45 0.70 | wrong one first' % '')
Ts = {'T .20': lambda d: 0.20, 'T .25': lambda d: 0.25, 'T .20/.30<1ms': lambda d: 0.30 if d < 48 else 0.20, 'T .25/.35<1ms': lambda d: 0.35 if d < 48 else 0.25, 'T .30': lambda d: 0.30}
for nt, T in Ts.items():
    for nm, rule in (('pm .5, r2 -.1', lambda c: abs(c['g']) >= T(c['d']) and c['pmin'] >= 0.5 and c['r2'] <= -0.1),
                     ('pm .5, r2 -.1, agree .6', lambda c: abs(c['g']) >= T(c['d']) and c['pmin'] >= 0.5 and c['r2'] <= -0.1 and c['agree'] >= 0.6),
                     ('6of6, r2 -.1', lambda c: abs(c['g']) >= T(c['d']) and c['n6'] >= 6 and c['r2'] <= -0.1),
                     ('5of6, r2 -.1', lambda c: abs(c['g']) >= T(c['d']) and c['n6'] >= 5 and c['r2'] <= -0.1),
                     ('5of6 + pm .5, r2 -.1', lambda c: abs(c['g']) >= T(c['d']) and c['n6'] >= 5 and c['pmin'] >= 0.5 and c['r2'] <= -0.1),
                     ('6of6 + pm .5, r2 -.2', lambda c: abs(c['g']) >= T(c['d']) and c['n6'] >= 6 and c['pmin'] >= 0.5 and c['r2'] <= -0.2),
                     ('5of6 + pm .5, r2 -.1, agree .5', lambda c: abs(c['g']) >= T(c['d']) and c['n6'] >= 5 and c['pmin'] >= 0.5 and c['r2'] <= -0.1 and c['agree'] >= 0.5)):
        o = rates(rule); print('%-16s %-40s %5.1f %5.1f %5.1f %5.1f %5.1f | %5.0f %5.0f %5.0f | %4.1f' % (nt, nm, *o))
