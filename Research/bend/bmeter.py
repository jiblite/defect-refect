# The bend meter, two ways: from the app's model file (Resources/bend.bin, written by Tools/export_bend_model.py) with
# plain numpy, and from the trainer's own file (.pt, with torch). Both give, for a sound's two half-waves, the squeeze
# per half-wave and level (dB) and the verdict (a logit).
import struct, numpy as np
NF = 184; NG = 11
class Bin:
    """The app's model file: one net or several, their answers averaged."""
    def __init__(s, path):
        d = open(path, 'rb').read(); assert d[:4] == b'DRBM', 'not a bend meter model file'; ver, nf, ng, count = struct.unpack_from('<IIII', d, 4); assert ver == 2 and nf == NF and ng == NG and 1 <= count <= 16; s.p = 20
        def u32(): v = struct.unpack_from('<I', d, s.p)[0]; s.p += 4; return v
        def arr(shape): n = u32(); assert n == int(np.prod(shape)); a = np.frombuffer(d, '<f4', n, s.p).reshape(shape); s.p += 4 * n; return a
        s.nets = []
        for _ in range(count):
            layers = u32(); mean = arr((NF,)); sd = arr((NF,)); lin = []; nin = 2 * NF + 1
            for _ in range(layers): out = u32(); lin.append((arr((out, nin)), arr((out,)))); nin = out
            assert nin == NG + 1; s.nets.append((mean, sd, lin))
        assert s.p == len(d)
    def run(s, X, ratio_db):
        """X (2, NF), the upper half-wave first; ratio_db: the upper half's peak against the lower half's -> (2, NG) squeeze in dB, the logit."""
        S = np.zeros((2, NG)); logit = 0.0
        for mean, sd, lin in s.nets:
            Z = ((np.asarray(X, np.float32) - mean) / sd).astype(np.float32)
            for h in range(2):
                a = np.concatenate([Z[h], Z[1 - h], np.array([(ratio_db if h == 0 else -ratio_db) / 20.0], np.float32)]).astype(np.float32)
                for i, (w, b) in enumerate(lin):
                    v = w.astype(np.float64) @ a.astype(np.float64) + b.astype(np.float64); a = (np.maximum(v, 0.0) if i + 1 < len(lin) else v).astype(np.float32)
                S[h] += 10.0 * a[:NG].astype(np.float64); logit += float(a[NG])
        return S / len(s.nets), logit / len(s.nets)
class Torch:
    """The trainer's file (btrain.py)."""
    def __init__(s, path):
        import torch, btrain; s.torch = torch; d = torch.load(path); s.net = btrain.Net(d['w']); s.net.load_state_dict(d['state']); s.net.eval()
    def run(s, X, ratio_db):
        with s.torch.no_grad(): S, lg = s.net(s.torch.tensor(np.asarray(X, np.float32)[None]), s.torch.tensor([float(ratio_db)], dtype=s.torch.float32))
        return S[0].numpy().astype(np.float64), float(lg[0])
class Several:
    """Several meters read as one: the mean of their answers."""
    def __init__(s, paths): s.m = [Torch(p) for p in paths]
    def run(s, X, ratio_db):
        r = [m.run(X, ratio_db) for m in s.m]; return np.mean([q[0] for q in r], 0), float(np.mean([q[1] for q in r]))
_have = {}
def model(what):
    if what not in _have: _have[what] = Bin(what) if what.endswith('.bin') else Several(what.split(',')) if ',' in what else Torch(what)
    return _have[what]
