# Per case: two result files of the same set side by side (only cases that differ by more than 1 point or in what is shown).
import sys, json
import long as L
a = {r['i']: r for r in map(json.loads, open(sys.argv[1]))}; b = {r['i']: r for r in map(json.loads, open(sys.argv[2]))}; thr = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
n = same = 0; dso = dsn = 0.0
for i in sorted(b):
    if i not in a: continue
    x, y = a[i], b[i]; n += 1; xs, ys = L.right(x), L.right(y); xo, yo = (x['out'] if xs else 0.0), (y['out'] if ys else 0.0); xn, yn = (x['onto'] if xs else 0.0), (y['onto'] if ys else 0.0); dso += yo - xo; dsn += yn - xn
    if xs == ys and abs(xo - yo) <= thr and abs(xn - yn) <= thr: same += 1; continue
    print('#%-3d %-10s %-11s %2d s  shown %d -> %d  score %6.2f -> %6.2f | out %6.1f -> %6.1f | onto %6.1f -> %6.1f | off %s -> %s' % (i, y['effect'], y['fam'], y['seconds'], xs, ys, x['score'] if x['found'] else float('nan'), y['score'] if y['found'] else float('nan'), xo, yo, xn, yn,
          '%.0f' % x['out_off'] if 'out_off' in x else '-', '%.0f' % y['out_off'] if 'out_off' in y else '-'))
print('%d cases compared, %d the same within %.1f; sum of changes: out %+.1f, onto %+.1f' % (n, same, thr, dso, dsn))
