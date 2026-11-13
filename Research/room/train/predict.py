# One door to the "room or sound" model in use: gains(ft) -> (T, 41) dB, 0 .. -30, for the features of a sound (common.features).
# MODEL = a name in this folder: 'unet_b' (neural, <name>.pt), 'gb_w.pkl' (the quick boosted-tree model), or 'first+second':
# two neural models, the second one looks again with the first one's answer and what the found room predicts (roomnum.second_look).
import os, sys, pickle
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np
NAME = os.environ.get('MODEL', 'unet_b'); _M = {}
def _load(nm):
    if nm not in _M:
        if nm.endswith('.pkl'): _M[nm] = ('gb', pickle.load(open('/tmp/claude-0/v2/train/' + nm, 'rb')))
        else:
            import torch, net; torch.set_num_threads(1); ck = torch.load('/tmp/claude-0/v2/train/%s.pt' % nm, map_location='cpu', weights_only=False); m = net.UNet(**ck['cfg']); m.load_state_dict(ck['state']); m.eval(); _M[nm] = ('unet', m)
    return _M[nm]
def gains(ft, name=None):
    nm = name or NAME
    if '+' in nm:
        import torch, net, roomnum; a, b = nm.split('+'); g1 = gains(ft, a); m = _load(b)[1]; LE = ft['LE'].astype(np.float32)
        x = np.concatenate([net.inputs(LE, ft['U64'].astype(np.float32), ft['U256'].astype(np.float32)), roomnum.second_look(ft['LE'].astype(np.float64), g1.astype(np.float64))])
        with torch.no_grad(): return m(torch.from_numpy(x.astype(np.float32))[None])[0].numpy().astype(np.float64)
    kind, m = _load(nm)
    if kind == 'gb':
        import gbfeat; X = gbfeat.features(ft['LE'], ft['U64'], ft['U256']); return np.clip(m.predict(X.reshape(-1, X.shape[-1])).reshape(ft['LE'].shape), -30.0, 0.0)
    import torch, net
    with torch.no_grad(): return m(torch.from_numpy(net.inputs(ft['LE'].astype(np.float32), ft['U64'].astype(np.float32), ft['U256'].astype(np.float32), m.cfg['cin'] == 6))[None])[0].numpy().astype(np.float64)       # 6 input planes = with the two fastest-fall planes
