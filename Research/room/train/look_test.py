# How far does the plain physical picture go? "What the room should add at every band and moment" (the direct sound smeared
# with the room's decay, times its strength) against what is there, turned into gains by simple rules, scored on the
# scoreboard cases at band level (net.proxy).  A: with the TRUE direct sound and TRUE room numbers.  B: with the first
# model's answer and the numbers found from it (needs pass2_sb/ from mkpass2.py).
import sys, glob, json, os
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, common as cm, net, roomnum
tr = np.load('/tmp/claude-0/v2/train/roomtruth_sb.npz', allow_pickle=True); names = list(tr['names'])
def smear_of(S, T60):
    P = np.zeros_like(S)
    for b in range(S.shape[1]):
        r = 10 ** (-6.0 * cm.DT / T60[b]) if np.isfinite(T60[b]) else 0.0; acc = 0.0
        for t in range(S.shape[0]): acc = r * acc + (1.0 - r) * S[t, b]; P[t, b] = acc
    return P
def rules(ratio_db):
    out = {'subtract': 10 * np.log10(np.clip(1.0 - 10 ** (ratio_db / 10.0), 1e-3, 1.0))}
    for th in (-9.0, -6.0, -3.0, 0.0):
        for s in (1.5, 3.0): out['step at %+.0f dB, width %.1f' % (th, s)] = -30.0 / (1.0 + np.exp(-(ratio_db - th) / s))
    return out
rows = {}
for f in sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz')):
    z = np.load(f); LE = z['LE'].astype(np.float64); LS = z['LS'].astype(np.float64); info = json.loads(str(z['info'])); i = names.index(os.path.basename(f)[:-4]); E = 10 ** (LE / 10.0); res = {}
    T60 = np.where(tr['ok'][i], tr['T60'][i], np.nan); kdb = np.where(tr['ok'][i], tr['kdb'][i], np.nan)
    if np.isfinite(T60).any():
        T60 = np.where(np.isfinite(T60), T60, np.nanmedian(T60)); kdb = np.where(np.isfinite(kdb), kdb, np.nanmedian(kdb)); P = smear_of(10 ** (LS / 10.0), T60)
        ratio = np.clip(10 * np.log10(10 ** ((kdb + roomnum.BUILD_UP_DB) / 10.0)[None, :] * P / np.maximum(E, 1e-30) + 1e-9), -30.0, 10.0)
        for k, g in rules(ratio).items(): res['A ' + k] = net.proxy(g, LE, LS)[0]
    p2 = f.replace('/sb/', '/pass2_sb/')
    if os.path.exists(p2):
        q = np.load(p2); g1 = q['g1'].astype(np.float64); X = q['X'].astype(np.float64); res['B first answer'] = net.proxy(g1, LE, LS)[0]
        if X[0].mean() > -0.99:
            ratio = (X[0] - 0.5) * 20.0
            for k, g in rules(ratio).items():
                res['B ' + k] = net.proxy(g, LE, LS)[0]; res['B min(first, %s)' % k] = net.proxy(np.minimum(g1, g), LE, LS)[0]; res['B mean(first, %s)' % k] = net.proxy(0.5 * (g1 + g), LE, LS)[0]
        else:
            for k in rules(np.zeros((2, 2))): res['B ' + k] = res['B min(first, %s)' % k] = res['B mean(first, %s)' % k] = res['B first answer']
    for k, v in res.items(): rows.setdefault(k, []).append((info['split'], info['drr'], v))
print('%-44s %s' % ('', 'dev: median mean <0 | unseen: median mean <0 | light  wet'))
for k, v in sorted(rows.items(), key=lambda kv: (kv[0][0], -np.median([x[2] for x in kv[1] if x[0] == 'dev']))):
    d = np.array([x[2] for x in v if x[0] == 'dev']); t = np.array([x[2] for x in v if x[0] == 'test']); a = np.array([x[2] for x in v]); dr = np.array([x[1] for x in v])
    print('%-44s %5.1f %5.1f %3.0f%% | %5.1f %5.1f %3.0f%% | %5.1f %5.1f  (n=%d)' % (k, np.median(d), np.mean(d), 100 * np.mean(d < 0), np.median(t), np.mean(t), 100 * np.mean(t < 0), np.median(a[dr >= 8]), np.median(a[dr < 0]), len(v)))
