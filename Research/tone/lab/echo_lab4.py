# Which test separates real reflections from a clean sound's own cepstral peaks? Uses the candidates collected by echo_lab3.py.
import json, numpy as np
z = json.load(open('work/echo3_clean.json')); files = z['files']; R = z['R']; kind = np.array([f.split('_')[1] for f in files])
# rows: per clip [clean candidates, (d, g, cands), (d, g, cands), (d, g, cands)]
def feats(c):   # d, g, g2, agree, med, p1, p2, p3
    d, g, g2, agree, med, p1, p2, p3 = c; parts = np.array([p1, p2, p3]); same = (np.sign(parts) == np.sign(g)).all(); pmin = np.abs(parts).min() if same else 0.0
    return dict(d=d, g=g, r2=g2 / (g * g + 1e-9), agree=agree, medr=med / g, pmin=pmin / abs(g))
clean = [feats(c) for r in R for c in r[0]]; true = []; missed = 0
for r in R:
    for d0, g0, cands in r[1:]:
        hit = [feats(c) for c in cands if abs(c[0] - d0) < 0.6]
        if hit: true.append(dict(hit[0], g0=g0))
        else: missed += 1
print('clean candidates (gain >= 0.08 somewhere in 0.5..15 ms): %d from %d clips; simulated reflections: %d found among the candidates, %d not' % (len(clean), len(R), len(true), missed))
def show(name, key, edges):
    a = np.array([c[key] for c in clean]); b = np.array([c[key] for c in true]); print('  %-34s' % name, ' | '.join('%s: clean %3.0f%% true %3.0f%%' % (('<%.2f' % e), 100 * (a < e).mean(), 100 * (b < e).mean()) for e in edges))
show('second peak / g^2 (reflection: -0.5)', 'r2', (-0.4, -0.2, -0.1, 0.0, 0.2)); show('frames agreeing', 'agree', (0.5, 0.6, 0.7, 0.8, 0.9)); show('median per-frame ripple / g', 'medr', (0.1, 0.2, 0.3, 0.5)); show('weakest third / g (0 if a sign differs)', 'pmin', (0.2, 0.4, 0.5, 0.6, 0.7))
print('gain of true ones vs set gain: median ratio %.2f, 10th..90th percentile %.2f..%.2f' % tuple(np.percentile([c['g'] / c['g0'] for c in true], [50, 10, 90])))
def rule(c, T, pm, r2, ag): return abs(c['g']) >= T(c['d']) and c['pmin'] >= pm and c['r2'] <= r2 and c['agree'] >= ag
def rates(T, pm, r2, ag):
    fa = np.array([any(rule(feats(c), T, pm, r2, ag) for c in r[0]) for r in R]); out = [100 * fa.mean()] + [100 * fa[kind == k].mean() for k in ('real-recording', 'fluid-melody', 'sample-loop', 'synthetic')]
    for j in range(3): out.append(100 * np.mean([any(abs(c[0] - r[j + 1][0]) < 0.6 and rule(feats(c), T, pm, r2, ag) for c in r[j + 1][2]) for r in R]))
    return out
print('rule: gain >= T(delay), weakest third >= pm * gain, second peak <= r2 * g^2, frames agreeing >= ag')
print('%-46s false alarms: all real melody loops synth | found g=0.30 0.45 0.70' % '')
for nameT, T in (('T=0.15', lambda d: 0.15), ('T=0.20', lambda d: 0.20), ('T=0.25', lambda d: 0.25), ('T=0.15, x2 under 1.5 ms', lambda d: 0.30 if d < 72 else 0.15), ('T=0.20, x2 under 1.5 ms', lambda d: 0.40 if d < 72 else 0.20), ('T=0.20, 0.3 under 1 ms', lambda d: 0.30 if d < 48 else 0.20)):
    for pm, r2, ag in ((0.0, 9.0, 0.0), (0.5, 9.0, 0.0), (0.0, -0.1, 0.0), (0.0, 9.0, 0.8), (0.5, -0.1, 0.0), (0.6, -0.2, 0.0), (0.5, 0.0, 0.0), (0.5, -0.1, 0.6)):
        o = rates(T, pm, r2, ag); print('%-24s pm %.1f r2 %+.1f ag %.1f   %5.1f %5.1f %5.1f %5.1f %5.1f | %5.0f %5.0f %5.0f' % (nameT, pm, r2, ag, *o))
