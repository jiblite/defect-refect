# How does the room engine cope when a sound has little difference between left and right? Scoreboard sounds with reverb,
# narrowed as a whole (side part x w), w = 1 (as they are) .. 0 (mono): is a room still found, how sure, and what a simple
# measure of "how much is not shared between the channels" reads (for the app to know when the engine cannot see).
#   MODEL=<name> python3 narrow_test.py [step=12]
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, engine, roomnum
MODEL = os.environ.get('MODEL', 'unet_b9'); step = int(sys.argv[1]) if len(sys.argv) > 1 else 12
C0, C1, C2 = [float(v) for v in os.environ.get('RULE', '1.847,0.860,1.165').split(',')]
def look(x):
    an = engine.analyse(x, MODEL); LE = an['LE']; g = an['g']; live = LE > LE.max() - 50.0; E = 10 ** (LE / 10)
    ft = cm.features(x); u = 10 ** (np.clip(ft['U256'], -40, 0) / 10); own = 10 * np.log10(max((E * u)[live].sum(), 1e-30) / E[live].sum())
    deep = float(np.mean(g[live] < -12.0)); bands = int((an['ev'] > 0).sum()) if an['found'] else 0
    sure = 1 / (1 + np.exp(-(C0 + C1 * np.log10(deep + 0.003) + C2 * bands / 41.0))) if an['found'] else 0.09
    return bool(an['found']), sure, deep, bands, own, (float(np.median(an['T60'][roomnum.MID])) if an['found'] else 0.0)
W = [1.0, 0.5, 0.25, 0.1, 0.03, 0.0]; rows = []
ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[3::step]
for d in ds:
    c = truth.load(d, 'A'); x = c['wet']; m = 0.5 * (x[:, 0] + x[:, 1]); s = 0.5 * (x[:, 0] - x[:, 1]); r = []
    for w in W: r.append(look(np.stack([m + w * s, m - w * s], 1)))
    rows.append(r); print('%-26s' % c['name'][:26], ' | '.join('%s %.2f own %5.1f T %.1f' % ('Y' if a[0] else '-', a[1], a[4], a[5]) for a in r), flush=True)
print('width          ', '  '.join('%6.2f' % w for w in W))
print('found on       ', '  '.join('%5.0f%%' % (100 * np.mean([r[i][0] for r in rows])) for i in range(len(W))))
print('rule > 65%     ', '  '.join('%5.0f%%' % (100 * np.mean([r[i][1] > 0.65 for r in rows])) for i in range(len(W))))
print('not shared, dB ', '  '.join('%6.1f' % np.median([r[i][4] for r in rows]) for i in range(len(W))))
