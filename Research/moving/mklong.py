# Long test sounds for the moving-tone engine: clean sounds of 30 to 48 s (scoreboard corpus sounds one after another), the
# same with an outside tool's phaser or filter sweep from end to end, and with the effect on the second half only.
#   python3 mklong.py            -> /dev/shm/mvl/*.wav and list.json (per case: dry, wet, the other sound and the truth on it)
import sys, os, json, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/detect/py']
import zoo_fx as Z
SR = 48000; OUT = '/dev/shm/mvl/'; r = np.random.default_rng(404); meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); ids = sorted(meta)
def rd(cid):
    x = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % cid, dtype='float64', always_2d=True)[0]; return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def chain(names, fade=2400):
    out = rd(names[0])
    for nm in names[1:]:
        x = rd(nm); w = np.linspace(0.0, 1.0, fade)[:, None]; out = np.concatenate([out[:-fade], out[-fade:] * (1 - w) + x[:fade] * w, x[fade:]])
    return out
def put(name, x): sf.write(OUT + name + '.wav', x, SR, subtype='FLOAT')
cases = []; clean = []
for i in range(12):
    names = [ids[int(k)] for k in r.choice(len(ids), int(r.integers(5, 9)), replace=False)]; x = chain(names); x *= min(1.0, 0.7 / np.abs(x).max()); put('clean%02d' % i, x); clean.append(x); cases.append(dict(name='clean%02d' % i, kind='clean', parts=names, sec=len(x) / SR))
TOOLS = dict(ph=['pha_caps', 'pha_pedalboard', 'pha_calf', 'pha_swh'], sw=['swp_caps', 'swp_mda', 'swp_pb_ladder', 'swp_pb_bandpass'])
for tag in ('ph', 'sw'):
    for k in range(8):
        tool = TOOLS[tag][k % 4]; x = clean[k]; o = clean[(k + 5) % 12]; best = None
        for _ in range(3):                                   # of three draws of the settings the one that does most to the sound
            prm = Z.ZOO[tool][2](r); w = Z.apply(tool, x, prm)[:len(x)]; d = float(np.sqrt(((w - x) ** 2).mean()) / np.sqrt((x ** 2).mean()))
            if best is None or d > best[0]: best = (d, prm, w)
        d, prm, w = best; truth = Z.apply(tool, o, prm)[:len(o)]; g = min(1.0, 0.98 / max(np.abs(w).max(), np.abs(truth).max(), 1e-9)); nm = '%sL%02d' % (tag, k)
        put(nm + '_wet', w * g); put(nm + '_truth', truth * g); cases.append(dict(name=nm, kind=tag, tool=tool, params=prm, dry='clean%02d' % k, other='clean%02d' % ((k + 5) % 12), gain=g, change=d, sec=len(x) / SR))
        if k < 4:                                            # the effect comes in half way (20 ms fade)
            n = len(x); h = n // 2; m = np.clip((np.arange(n) - h) / 960.0, 0.0, 1.0)[:, None]; hw = x * (1 - m) + w * m; nm = '%sH%02d' % (tag, k); put(nm + '_wet', hw * g)
            cases.append(dict(name=nm, kind=tag + 'half', tool=tool, params=prm, dry='clean%02d' % k, gain=g, from_s=h / SR, sec=n / SR))
        print(nm, tool, '%.1f s, changes the sound by %.2f' % (len(x) / SR, d), flush=True)
json.dump(cases, open(OUT + 'list.json', 'w'), default=float)
