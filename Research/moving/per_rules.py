# Rules for the curve's period, compared on the kept curves (per.py, per_long.py) against the effect's own rate.
import json, numpy as np
R = json.load(open('/dev/shm/per/list.json')) + json.load(open('/dev/shm/per/list_long.json'))
def corr(G, cap=None):
    T = len(G); n = T // 2 + 1; e = (G ** 2).sum(1); r = np.zeros(n); D = G - G.mean(0); ed = (D ** 2).sum(1); rd = np.zeros(n)
    for L in range(12, n):
        r[L] = (G[:-L] * G[L:]).sum() / max(np.sqrt(e[:-L].sum() * e[L:].sum()), 1e-12); rd[L] = (D[:-L] * D[L:]).sum() / max(np.sqrt(ed[:-L].sum() * ed[L:].sum()), 1e-12)
    return r, rd
def refine(r, L): d = r[L - 1] - 2 * r[L] + r[L + 1]; return L + (0.5 * (r[L - 1] - r[L + 1]) / d if d < 0 else 0.0)
def now(r, rd):
    for L in range(13, len(r) - 1):
        if r[L] >= 0.8 and r[L] >= r[L - 1] and r[L] >= r[L + 1]: return refine(r, L)
    return 0.0
def new(r, rd, moves=0.2, near=0.1):
    c = [L for L in range(13, len(r) - 1) if r[L] >= 0.8 and r[L] >= r[L - 1] and r[L] >= r[L + 1] and rd[L] >= moves]
    if not c: return 0.0
    best = max(rd[L] for L in c); return refine(r, min(L for L in c if rd[L] >= best - near))
def dm(r, rd, thr=0.5, near=0.1):
    c = [L for L in range(13, len(r) - 1) if rd[L] >= thr and rd[L] >= rd[L - 1] and rd[L] >= rd[L + 1]]
    if not c: return 0.0
    best = max(rd[L] for L in c); return refine(rd, min(L for L in c if rd[L] >= best - near))
def truth(t, p):
    if t.startswith('swp_pb'): return 1 / p['rate']
    if t == 'swp_mda': return 1 / 10 ** (3 * p['lfo_rate'] - 1.5)
    if t == 'pha_calf': return 1 / p['mod_rate']
    if t == 'pha_pedalboard': return 1 / p['rate_hz']
    if t == 'pha_swh': return 1 / p['rate']
    return float('nan')
def verdict(v, tr, T):
    if not np.isfinite(tr): return 'none' if v == 0 else 'any'
    if v == 0: return 'zero' if tr <= T / 2 else 'zero-ok'
    k = v / tr; m = round(k)
    return 'right' if abs(k - 1) < 0.08 else ('multiple' if m >= 2 and abs(k - m) < 0.08 * m else 'WRONG')
tot = {}
for q in R:
    G = np.load('/dev/shm/per/%s.npy' % q['name']); T = len(G); tr = truth(q['tool'], q['params']) * 46.875; r, rd = corr(G); out = []
    for name, f in (('now', now), ('new', new), ('new.3', lambda r, rd: new(r, rd, 0.3)), ('dm.5', dm), ('dm.4', lambda r, rd: dm(r, rd, 0.4))):
        v = f(r, rd); vd = verdict(v, tr, T); tot.setdefault(name, {}).setdefault(vd, 0); tot[name][vd] += 1; out.append('%s %6.1f %-8s' % (name, v, vd))
    print('%-22s %-11s sure %.2f T %4d truth %6.1f | %s' % (q['name'], q['kind'], q['sure'], T, tr, ' | '.join(out)))
for k, v in tot.items(): print(k, dict(sorted(v.items())))
