# The curve reader's nets, two ways: from the app's model file (Resources/filter.bin, written by
# Tools/export_reader_model.py) with plain numpy, and from the trainer's own files (.pt, with torch). Both give, for a
# sound's six curves, every net's curve over the bands (dB) and how sure it is (a logit).
import struct, numpy as np
NBAND = 104
class Bin:
    """The app's model file."""
    def __init__(s, path):
        d = open(path, 'rb').read(); assert d[:4] == b'DRCR', 'not a curve reader model file'; ver, count = struct.unpack_from('<II', d, 4); assert ver == 2; s.p = 12; s.nets = []
        def u32(): v = struct.unpack_from('<I', d, s.p)[0]; s.p += 4; return v
        def arr(shape): n = u32(); assert n == int(np.prod(shape)); a = np.frombuffer(d, '<f4', n, s.p).astype(np.float64).reshape(shape); s.p += 4 * n; return a
        for _ in range(count):
            nin, nb, layers, ch, width = (u32() for _ in range(5)); assert nb == NBAND; conv = []
            for i in range(layers): dil = u32(); w = arr((ch, nin if i == 0 else ch, width)); b = arr((ch,)); conv.append((dil, w, b))
            s.nets.append(dict(conv=conv, cw=arr((ch,)), cb=arr((1,)), tw=arr((2 * ch,)), tb=arr((1,))))
        assert s.p == len(d)
    def run(s, X):
        """X (6, bands) in dB -> (nets, bands) curves in dB, (nets,) logits."""
        X = np.asarray(X, np.float32); C, T = [], []                 # the app works on single-precision curves and adds up in double precision
        for n in s.nets:
            h = np.concatenate([X.astype(np.float32) / np.float32(20.0) + np.float32(2.0), np.linspace(-1, 1, NBAND)[None].astype(np.float32)], 0).astype(np.float32)
            for dil, w, b in n['conv']:
                half = w.shape[2] // 2; pad = half * dil; hp = np.pad(h, ((0, 0), (pad, pad))); o = np.repeat(b.astype(np.float32)[:, None], NBAND, 1)
                for j in range(w.shape[2]): o = o + (w[:, :, j].astype(np.float32) @ hp[:, j * dil:j * dil + NBAND]).astype(np.float32)
                h = np.maximum(o, 0).astype(np.float32)
            h = h.astype(np.float64); C.append(20.0 * (n['cw'] @ h + n['cb'][0])); T.append(float(n['tw'] @ np.concatenate([h.mean(1), h.max(1)]) + n['tb'][0]))
        return np.stack(C), np.array(T)
class Torch:
    """The trainer's files (ftrain.py)."""
    def __init__(s, paths):
        import torch, ftrain; s.torch = torch; s.nets = []
        for p in ([paths] if isinstance(paths, str) else paths):
            sd = torch.load(p); n = ftrain.Net(sd['c.0.weight'].shape[0], sd['c.0.weight'].shape[1] - 1); n.load_state_dict(sd); n.eval(); s.nets.append(n)
    def run(s, X):
        xt = s.torch.tensor(np.asarray(X)[None].astype(np.float32))
        with s.torch.no_grad(): outs = [n(xt) for n in s.nets]
        return np.stack([o[0][0].numpy().astype(np.float64) for o in outs]), np.array([float(o[1][0]) for o in outs])
_have = {}
def model(what):
    """A model file (.bin), one trainer's file (.pt) or a list of them -> the nets, kept for the next call."""
    key = what if isinstance(what, str) else tuple(what)
    if key not in _have: _have[key] = Bin(what) if isinstance(what, str) and what.endswith('.bin') else Torch(what)
    return _have[key]
