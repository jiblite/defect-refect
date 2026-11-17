# What the stretches of a long sound say about where the movement is (the windows and spans drlevel reports), set
# against where the effect really is. python3 win.py file.jsonl [...] [price=3 lean=0.15 ...]
import sys, json, numpy as np
SR = 48000

def dp(W, price, lean):
    """The engine's choice of 'there' for every stretch (spansOf in Level.h, without the fine placing of the changes)."""
    m = len(W)
    if m == 0: return []
    b0 = np.zeros(m); b1 = np.zeros(m); f0 = np.zeros(m, int); f1 = np.ones(m, int)
    for k, (a, b, u, w, own, on) in enumerate(W):
        g = w * (u + (lean if own else -lean))
        if k == 0: b1[0] = g; continue
        f0[k] = b1[k - 1] - price > b0[k - 1]; b0[k] = max(b0[k - 1], b1[k - 1] - price)
        f1[k] = b1[k - 1] >= b0[k - 1] - price; b1[k] = max(b1[k - 1], b0[k - 1] - price) + g
    st = int(b1[-1] > b0[-1]); out = [0] * m
    for k in range(m - 1, -1, -1):
        out[k] = st
        if k > 0: st = f1[k] if st else f0[k]
    return out

def table(rows, price=None, lean=None, tag=''):
    for fam in ('tremolo', 'autopan', 'gatedrhythm', 'all'):
        rs = [r for r in rows if r['found'] and (fam == 'all' or r['fam'] == fam)]
        if not rs: continue
        live_t = dead_t = live_on = dead_on = 0.0
        for r in rs:
            on = dp(r['windows'], price, lean) if price is not None else [w[5] for w in r['windows']]
            for w, lv, o in zip(r['windows'], r['win_live'], on):
                d = (w[1] - w[0]) / SR; live_t += d * lv; dead_t += d * (1 - lv); live_on += d * lv * o; dead_on += d * (1 - lv) * o
        print('  %s%-12s: with the effect %6.0f s, marked %5.1f%% | without it %6.0f s, marked %5.1f%%' % (tag, fam, live_t, 100 * live_on / max(live_t, 1e-9), dead_t, 100 * dead_on / max(dead_t, 1e-9)))

if __name__ == '__main__':
    files = [a for a in sys.argv[1:] if '=' not in a]; opts = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
    for path in files:
        rows = sorted((json.loads(l) for l in open(path)), key=lambda r: r['i']); rows = [r for r in rows if r.get('windows')]
        print(path, len(rows), 'sounds with a movement')
        # what single stretches say
        for own in (1, 0):
            for name, sel in (('with the effect', lambda lv: lv >= 0.95), ('without it', lambda lv: lv <= 0.05)):
                for fam in ('tremolo', 'autopan', 'gatedrhythm'):
                    u = np.array([w[2] for r in rows if r['found'] and r['fam'] == fam for w, lv in zip(r['windows'], r['win_live']) if w[4] == own and sel(lv) and w[3] > 0])
                    n0 = sum(1 for r in rows if r['found'] and r['fam'] == fam for w, lv in zip(r['windows'], r['win_live']) if w[4] == own and sel(lv) and w[3] == 0)
                    if len(u): print('  stretches in pieces that %s, %-15s %-12s: %5d (+%4d that say nothing)  u: 5%% %5.2f  25%% %5.2f  median %5.2f  75%% %5.2f  95%% %5.2f | u > 0: %3.0f%%' % ('carry it by the rule' if own else 'do not            ', name, fam, len(u), n0, *np.percentile(u, [5, 25, 50, 75, 95]), 100 * (u > 0).mean()))
        print(' as the engine marked them:'); table(rows)
        if 'price' in opts:
            for price in [float(v) for v in opts['price'].split(',')]:
                for lean in [float(v) for v in opts.get('lean', '0.15').split(',')]: table(rows, price, lean, 'price %.1f lean %.2f ' % (price, lean))
        old = {}
        if 'before' in opts: old = {r['i']: r for r in map(json.loads, open(opts['before']))}
        for r in rows:
            if not r['found'] or 'out_off' not in r: continue
            W = r['windows']; lv = np.array(r['win_live']); first = lv[0] > 0.5; k = [i for i in range(len(W)) if 0.02 < lv[i] < 0.98]
            if k: i = k[0]; a, b = W[i][0], W[i][1]; border = (a + lv[i] * (b - a)) if first else (a + (1 - lv[i]) * (b - a))
            else: border = W[[j for j in range(1, len(W)) if (lv[j] > 0.5) != (lv[j - 1] > 0.5)][0]][0]
            border /= SR; sp = r['spans']; mk = (max((b for a, b in sp), default=0) if first else min((a for a, b in sp), default=r['seconds'] * SR)) / SR; err = (mk - border) if first else (border - mk); o = old.get(r['i'], {})
            print('    #%-3d %-16s %2d s %-8s %5.2f Hz %5.1f dB score %5.2f | effect %-5s border %6.2f marked %-22s off by %+6.2f s | out %6.1f (%6.1f) | other half %7.1f dB (%7.1f)' % (r['i'], r['effect'], r['seconds'], r['kind'], r['rate'], r['depth'], r['score'], 'first' if first else 'last', border,
                  ' '.join('%.1f-%.1f' % (a / SR, b / SR) for a, b in sp), err, r['out'], o.get('out', float('nan')), r['out_off'], o.get('out_off', float('nan'))))
