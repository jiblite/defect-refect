# The rule "a room must still be there with the echoes taken out", checked where it could do harm:
#   A  the scoreboard's reverb sounds on which the echo engine finds an echo (mostly true early reflections of the reverb)
#   B  reverb sounds with a synthetic echo put on top: the room is real and must stay
# Per sound the room engine's sureness: as it is, and with the found echoes out (drecho out). B also: before the echo was put on.
import sys, os, re, glob, json, zlib, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '.')
B = '/tmp/claude-0/drbuild/'; RES = '/home/claude/defect-refect/Resources'; D = '/dev/shm/ep/'; S = '/tmp/claude-0/v2/bench/sets/reverb/'
fn = 'roomcheck2.jsonl'; done = {json.loads(l)['name'] for l in open(fn)} if os.path.exists(fn) else set(); out = open(fn, 'a')
def verdict(path):
    r = subprocess.run(['nice', B + 'drrender', RES, path, '-', '-'], capture_output=True, text=True); m = re.search(r'room: found=(\d) detector=([0-9.e+-]+) sureness=([0-9.e+-]+)', r.stdout)
    e = re.search(r'echo: sure=([0-9.e+-]+)(.*)', r.stdout)
    return dict(room=float(m.group(3)), echo=float(e.group(1)), lag=e.group(2).strip()) if m and e else None
def without(path):
    subprocess.run(['nice', B + 'drecho', 'out', path, D + '_rc2o.wav'], capture_output=True, text=True); return verdict(D + '_rc2o.wav')
# A
st = [json.loads(l) for f in sorted(glob.glob('statscpp_*.jsonl')) for l in open(f)]
for s in st:
    if s.get('tag') != 'reverb' or 'error' in s or not any(c['found'] for c in s['ch']): continue
    name = 'A:' + s['name']
    if name in done: continue
    x, sr = sf.read(S + s['name'][:-2] + '/' + s['name'][-1] + '_wet.flac', dtype='float32', always_2d=True); sf.write(D + '_rc2.wav', x[:, :2], 48000, subtype='FLOAT')
    a = verdict(D + '_rc2.wav'); b = without(D + '_rc2.wav'); out.write(json.dumps(dict(name=name, wet=a, without=b)) + '\n'); out.flush(); print(name, a, b, flush=True)
# B
import scipy.signal as ss
def put(x, Dl, g, f):
    y = x.copy(); k = 1
    while k * Dl < len(x) and abs(g) * (f ** (k - 1) if f > 0 else 1.0) > 0.003:
        y[k * Dl:] += g * (f ** (k - 1) if f > 0 else 1.0) * x[:len(x) - k * Dl]; k += 1
        if f == 0.0: break
    return y
cases = sorted(os.listdir(S)); rng0 = np.random.default_rng(7); pick = [cases[i] for i in sorted(rng0.choice(len(cases), 14, replace=False))]
for i, c in enumerate(pick):
    name = 'B:' + c
    if name in done: continue
    x, sr = sf.read(S + c + '/A_wet.flac', dtype='float64', always_2d=True); x = x[:, :2]
    rng = np.random.default_rng(zlib.crc32(c.encode())); Dl = int(round(np.exp(rng.uniform(np.log(0.06), np.log(0.7))) * 48000)); g = float(rng.uniform(0.3, 0.7)); f = float(rng.uniform(0.2, 0.6)) if i % 2 else 0.0
    sf.write(D + '_rc2.wav', x.astype(np.float32), 48000, subtype='FLOAT'); a0 = verdict(D + '_rc2.wav')
    y = np.stack([put(x[:, 0], Dl, g, f), put(x[:, 1], Dl, g, f)], 1); pk = np.abs(y).max(); y *= min(1.0, 0.99 / pk)
    sf.write(D + '_rc2.wav', y.astype(np.float32), 48000, subtype='FLOAT'); a = verdict(D + '_rc2.wav'); b = without(D + '_rc2.wav')
    out.write(json.dumps(dict(name=name, before=a0, wet=a, without=b, par=dict(ms=Dl / 48.0, g=g, fb=f))) + '\n'); out.flush(); print(name, 'echo %.0f ms g %.2f fb %.2f' % (Dl / 48.0, g, f), a0, a, b, flush=True)
