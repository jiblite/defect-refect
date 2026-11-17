# The app's detection in Python, for bulk work: the same pictures (feat.py of the first prototype, which the C++ was ported
# from), the same CNNs and deciders read straight from the app's model files (Resources/models.bin, models27.bin), the same
# hand checks (the prototype's detect.py and echo.py), the same rules (highest of the 3 s windows, whole-file echo rule,
# Estimate.h mergeNew as it was up to 0.4.1). Against the C++ on 494 sounds: largest difference of a final score 0.0016, no
# shown / not shown decision differs.
import sys, struct, os
import paths; sys.path.insert(0, paths.PROTO)
import numpy as np, torch, torch.nn.functional as TF
from numba import njit
import feat as _feat, detect as _detect, echo as _echo
torch.set_num_threads(1)
SR = 48000; RES = paths.RES
OLD = ['flanger', 'chorus', 'phaser', 'haas', 'width', 'delay', 'reverb', 'distortion', 'bitcrush', 'filtersweep', 'tremolo', 'autopan']
NEW = ['saturation', 'exciter', 'ringmod', 'filter', 'stutter', 'gatedrhythm']
SHOWN = [f for f in OLD if f != 'reverb'] + NEW            # the 17 families this work is about (reverb has its own engine now)
NODE = np.dtype([('feat', '<i4'), ('thr', '<f4'), ('left', '<i4'), ('right', '<i4'), ('leaf', 'u1'), ('miss', 'u1'), ('value', '<f4')])

class _Rd:
    def __init__(s, b): s.b = b; s.p = 0
    def u32(s): v = struct.unpack_from('<I', s.b, s.p)[0]; s.p += 4; return v
    def f32a(s): n = s.u32(); a = np.frombuffer(s.b, '<f4', n, s.p).copy(); s.p += 4 * n; return a
    def str(s): n = s.u32(); v = s.b[s.p:s.p + n].decode(); s.p += n; return v
    def trees(s):
        base = s.f32a(); it = s.u32(); out = []
        for _ in range(it * len(base)):
            n = s.u32(); out.append(np.frombuffer(s.b, NODE, n, s.p).copy()); s.p += NODE.itemsize * n
        return base, out

@njit(cache=True)
def _boost(feat, thr, left, right, leaf, miss, value, start, x):
    o = 0.0
    for t in range(len(start) - 1):
        n = start[t]
        while leaf[n] == 0:
            v = x[feat[n]]
            if np.isnan(v): n = start[t] + (left[n] if miss[n] else right[n])
            else: n = start[t] + (left[n] if v <= thr[n] else right[n])
        o += value[n]
    return o

class Model:
    def __init__(s, path):
        r = _Rd(open(path, 'rb').read()); assert r.b[:4] == b'DRFX'; r.p = 4; r.u32()
        s.conv = []
        for _ in range(r.u32()): o, i = r.u32(), r.u32(); w = r.f32a(); b = r.f32a(); s.conv.append((torch.tensor(w.reshape(o, i, 3, 3)), torch.tensor(b)))
        s.pools = [(r.u32(), r.u32()) for _ in range(r.u32())]
        s.t = []
        for _ in range(2): o, i, k, pad, dil = r.u32(), r.u32(), r.u32(), r.u32(), r.u32(); w = r.f32a(); b = r.f32a(); s.t.append((torch.tensor(w.reshape(o, i, k)), torch.tensor(b), pad, dil))
        s.lin = []
        for _ in range(2): o, i = r.u32(), r.u32(); w = r.f32a(); b = r.f32a(); s.lin.append((torch.tensor(w.reshape(o, i)), torch.tensor(b)))
        s.fx = [r.str() for _ in range(r.u32())]
        s.dec = []
        for _ in range(r.u32()):
            r.str(); base, trees = r.trees(); assert len(base) == 1
            a = np.concatenate(trees); start = np.cumsum([0] + [len(t) for t in trees]).astype(np.int64)
            s.dec.append((float(base[0]), tuple(np.ascontiguousarray(a[k]) for k in ('feat', 'thr', 'left', 'right', 'leaf', 'miss', 'value')) + (start,)))
    def cnn(s, F):
        """F: (B, 4, 256, T) float32 pictures -> logits (B, n), reg (B, n), emb (B, 384)."""
        with torch.no_grad():
            x = torch.as_tensor(F, dtype=torch.float32); x = 0.5 * (x[:, :, 0::2] + x[:, :, 1::2])
            for b, (pf, pt) in enumerate(s.pools):
                for j in range(2): w, bb = s.conv[2 * b + j]; x = torch.relu(TF.conv2d(x, w, bb, padding=1))
                x = TF.max_pool2d(x, (pf, pt))
            B, C, H, W = x.shape; x = x.reshape(B, C * H, W)
            for w, bb, pad, dil in s.t: x = torch.relu(TF.conv1d(x, w, bb, padding=pad, dilation=dil))
            emb = torch.cat([x.mean(2), x.amax(2)], 1)
            return (emb @ s.lin[0][0].T + s.lin[0][1]).numpy(), (emb @ s.lin[1][0].T + s.lin[1][1]).numpy(), emb.numpy()
    def decide(s, x):
        """x: (n logits + 15 hand,) float32 -> the deciders' scores (n,)."""
        x = np.ascontiguousarray(x, np.float32); return np.array([1.0 / (1.0 + np.exp(-(base + _boost(*t, x)))) for base, t in s.dec])

_M = {}
def models():
    if not _M: _M['m12'] = Model(RES + 'models.bin'); _M['m27'] = Model(RES + 'models27.bin')
    return _M['m12'], _M['m27']

def hand(y):
    sc, ms = _echo.echo_detect(y, zmin=15, smin=3)
    st = _detect.stereo(y); nl = _detect.nonlinear(y); tr = _detect.tremolo(y)
    return [sc, ms or 0, st['side_ratio'], st['corr'], st.get('haas_z', 0), abs(st.get('haas_ms', 0)), st.get('pan_r2', 0), st.get('pan_span_db', 0),
            nl['clip_frac'], np.log10(nl['unique_frac'] + 1e-6), nl['hold'], nl['crest'], nl['kurtosis'], tr['r2'], tr['depth_db']]

def windows(y, W=3 * SR):
    if len(y) <= W: return [(0, np.vstack([y, np.zeros((W - len(y), 2))]))]
    return [(s, y[s:s + W]) for s in range(0, len(y) - W + 1, W // 2)]

def cut_for(f, cut=0.65): return min(0.99, max(cut + 0.15, 0.8 if cut == 0.65 else 0.0)) if f in NEW else cut

def analyze(y, emb=False):
    """y: (n, 2) float64 at 48 kHz. Everything the app's detection computes, per window and for the sound."""
    m12, m27 = models(); wins = [(s, w) for s, w in windows(y) if np.sqrt(np.mean(w ** 2)) >= 1e-4]
    out = dict(starts=[s for s, _ in wins])
    sc, ms = _echo.echo_detect(y); out['echo'] = (float(sc), float(ms or 0))
    if not wins:
        out.update(final={f: 0.0 for f in OLD + NEW}, hand=np.zeros((0, 15), np.float32), logit12=np.zeros((0, 12), np.float32), logit27=np.zeros((0, 27), np.float32), p12=np.zeros((0, 12)), p27=np.zeros((0, 27))); return out
    F = np.stack([_feat.features(w).astype(np.float32) for _, w in wins]); H = np.array([hand(w) for _, w in wins], np.float32)
    for key, m in (('12', m12), ('27', m27)):
        lg, rg, em = m.cnn(F); P = np.array([m.decide(np.r_[lg[i], H[i]]) for i in range(len(wins))])
        out['logit' + key] = lg; out['p' + key] = P
        if emb: out['emb' + key] = em
    out['hand'] = H
    s12 = dict(zip(m12.fx, out['p12'].max(0))); s27 = dict(zip(m27.fx, out['p27'].max(0)))
    for s in (s12, s27):
        if sc > 60 and s['bitcrush'] < 0.5: s['delay'] = max(s['delay'], min(0.99, 0.6 + sc / 1000.0))
    final = dict(s12); final.update({f: s27[f] for f in NEW})                 # Estimate.h mergeNew
    if any(final[f] > 0.8 for f in NEW):
        for f in OLD:
            if s27[f] < 0.15: final[f] = min(final[f], s27[f])
    out['s12'] = s12; out['s27'] = s27; out['final'] = {k: float(v) for k, v in final.items()}
    return out

def shown(final, cut=0.65): return [f for f in SHOWN if final[f] > cut_for(f, cut)]
