# writes the sounds of flong.py's cases to /dev/shm/fl: python3 fmk.py wet|part i [i ...]   (a = wet, d = dry, b = second dry, t = second wet)
import sys, os, json, numpy as np, soundfile as sf
import flong as F
sys.path.insert(0, '/tmp/claude-0/v2/detect/py')
import zoo_fx
L = F.L; SR = 48000
mode = sys.argv[1]; want = [int(v) for v in sys.argv[2:]]
jobs = [(mode, 6 * e + j, name, F.LENGTHS[j % 3], j < 3) for e, name in enumerate(F.EFFECTS) for j in range(6)]
for mode, i, name, seconds, same in jobs:
    if i not in want: continue
    rng = np.random.default_rng(L.SEED + dict(wet=29000, part=31000)[mode] + i); n = int(seconds * SR)
    par = zoo_fx.ZOO[name][2](rng); live = np.ones(n); info = dict(i=i, effect=name, seconds=seconds, par={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in par.items()})
    if mode == 'part':
        live = np.zeros(n); cut = int(n * rng.uniform(0.4, 0.6)); X = int(0.05 * SR)
        if rng.random() < 0.5: live[:cut] = 1; live[cut:cut + X] = np.linspace(1, 0, X); info['live'] = [0.0, cut / SR]
        else: live[cut:] = 1; live[cut - X:cut] = np.linspace(0, 1, X); info['live'] = [cut / SR, seconds]
    def fn(x): w = zoo_fx.apply(name, x, par); m = live[:len(x), None]; return x * (1 - m) + w * m
    dry, kind, used = L.long_sound(rng, seconds, same); dryb, kindb, usedb = L.long_sound(rng, seconds, same); wet = fn(dry); wetb = fn(dryb)
    p = '/dev/shm/fl/%s_%d_' % (mode, i)
    sf.write(p + 'a.wav', wet, SR, subtype='FLOAT'); sf.write(p + 'd.wav', dry, SR, subtype='FLOAT'); sf.write(p + 'b.wav', dryb, SR, subtype='FLOAT'); sf.write(p + 't.wav', wetb, SR, subtype='FLOAT')
    info['sound'] = kind; json.dump(info, open(p + 'info.json', 'w')); print(json.dumps(info))
