# Long and short test sounds for the chorus engine's pieces: several corpus sounds in a row with one chorus over all of it,
# and a sound whose chorus is switched on half way.
#   writes /dev/shm/chl/<name>_{dry,wet}.wav
import sys, os, json, numpy as np, soundfile as sf, pedalboard as pb
SR = 48000; OUT = '/dev/shm/chl/'; CORPUS = '/tmp/claude-0/v2/corpus/'; os.makedirs(OUT, exist_ok=True)
def chorus(x, p, pre=1.5):
    k = int(pre * SR); xin = np.vstack([np.zeros((k, 2)), x]); return pb.Pedalboard([pb.Chorus(**p)])(xin.T.astype(np.float32), SR, reset=True).T.astype(np.float64)[k:]
def dry(name):
    x = sf.read(CORPUS + 'dry/' + name + '.wav', always_2d=True)[0].astype(np.float64); return x if x.shape[1] > 1 else np.repeat(x, 2, 1)
meta = json.load(open(CORPUS + 'meta.json')); ids = [i for i in sorted(meta) if meta[i].get('kind') != 'real-recording' and not i.startswith('real')]
rng = np.random.default_rng(5); pick = list(rng.permutation(len(ids)))
def row(k): return np.concatenate([dry(ids[pick.pop()])[:, :2] for _ in range(k)])
def put(name, x, y): g = min(1.0, 0.98 / max(np.abs(y).max(), 1e-9)); sf.write(OUT + name + '_dry.wav', x, SR, subtype='FLOAT'); sf.write(OUT + name + '_wet.wav', y * g, SR, subtype='FLOAT'); print(name, '%.1f s' % (len(x) / SR), flush=True)
p = dict(rate_hz=0.6, depth=0.3, centre_delay_ms=9.0, feedback=0.0, mix=0.4)
a = row(7); put('longA', a, chorus(a, p))                                 # 42 s, the chorus on all of it
b = row(5)[:int(27.3 * SR)]; put('longB', b, chorus(b, p))                # 27.3 s, for putting A's chorus onto another length
c = row(6); w = chorus(c, p); k = int(17.0 * SR); f = np.clip((np.arange(len(c)) - k) / (0.05 * SR), 0.0, 1.0)[:, None]; put('halfC', c, c * (1 - f) * (1 - p['mix']) + (f * w + (1 - f) * 0) + 0 * c if False else (1 - f) * c * (1 - p['mix']) + f * w)      # 36 s, the chorus only from 17 s on (before it the sound at the level it has inside the chorus)
d = row(1)[:int(2.5 * SR)]; put('shortD', d, chorus(d, dict(rate_hz=1.2, depth=0.25, centre_delay_ms=8.0, feedback=0.0, mix=0.4)))      # 2.5 s
e = row(3)[:int(13.0 * SR)]; put('edgeE', e, chorus(e, p))                # 13 s: just over one piece
