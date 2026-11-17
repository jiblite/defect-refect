# Scores of the level engine on the scoreboard's cases, both ways round (found on A: out of A, onto B; then A and B swapped).
#   python3 ev.py [family ...] [--shift]      results: ev_<tag>.jsonl (one line per case and direction)
import sys, os, json, time, numpy as np
sys.path += ['/tmp/claude-0/v2/proto', '/tmp/claude-0/v2/bench', '/tmp/claude-0/v2/level']
import truth, score, level
S = '/tmp/claude-0/v2/bench/sets/'; args = [a for a in sys.argv[1:] if not a.startswith('--')]; fams = args or ['tremolo', 'autopan', 'gatedrhythm']
TAG = os.environ.get('TAG', 'v0'); out = open('ev_%s.jsonl' % TAG, 'a'); rows = []
def true_rate(p): return p.get('hz') or p.get('freq') or p.get('speed') or p.get('rate')
for fam in fams:
    for name in sorted(os.listdir(S + fam)):
        cA = truth.load(S + fam + '/' + name + '/', 'A'); cB = truth.load(S + fam + '/' + name + '/', 'B')
        for tag, c, o in (('A', cA, cB), ('B', cB, cA)):
            t0 = time.time(); info = {}; F = level.find(c['wet'], info=info); dt = time.time() - t0; tr = true_rate(c['meta']['params'])
            row = dict(fam=fam, name=name, dir=tag, split=c['meta']['split'], found=F is not None, true_rate=tr, secs=dt)
            if F is None: row.update(out=0.0, onto=0.0); desc = 'not found'
            else:
                y = level.apply(c['wet'], F, -1.0); z = level.apply(o['dry'], F, 1.0); row.update(out=score.removed(c['dry'], c['wet'], y)[0], onto=score.removed(o['wet'], o['dry'], z)[0], kind=F['kind'], rate=F['rate'], depth=F['depth'])
                if '--shift' in sys.argv and row['onto'] < 90:          # right movement, slightly off in time? the best of a few small shifts of the curve
                    sc = score.Scorer(o['wet']); d0 = sc(o['dry']); g = level.gain_db(F, len(o['dry']) + 4800); best = (row['onto'], 0)
                    for ms in (-20, -10, -5, -2, 2, 5, 10, 20):
                        k = int(ms * 48); gg = g[max(0, k):len(o['dry']) + max(0, k)] if k >= 0 else np.vstack([np.repeat(g[:1], -k, 0), g[:len(o['dry']) + k]])
                        v = 100 * (1 - sc(o['dry'] * 10 ** (gg[:len(o['dry'])] / 20)) / d0)
                        if v > best[0]: best = (v, ms)
                    row.update(onto_shift=best[0], shift_ms=best[1])
                desc = '%s %.3f Hz (true %s) depth %.1f dB' % (F['kind'], F['rate'], '%.3f' % tr if tr else '?', F['depth'])
            rows.append(row); out.write(json.dumps(row) + '\n'); out.flush()
            print('%-12s %-20s %s: out %6.1f onto %6.1f %s | %s (%.1fs)' % (fam, name, tag, row['out'], row['onto'], ('(best shift %+d ms: %.1f)' % (row['shift_ms'], row['onto_shift'])) if 'onto_shift' in row else '', desc, dt), flush=True)
print()
for fam in fams:
    r = [x for x in rows if x['fam'] == fam]; o = np.array([x['out'] for x in r]); t = np.array([x['onto'] for x in r])
    print('%-12s found %2d/%2d | out median %5.1f mean %5.1f worse %d | onto median %5.1f mean %5.1f worse %d' % (fam, sum(x['found'] for x in r), len(r), np.median(o), o.mean(), (o < 0).sum(), np.median(t), t.mean(), (t < 0).sum()))
