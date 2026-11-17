"""Export the trained moving-tone model (the small network that reads a moving filter's tone curve off the band picture
of a sound: PHASER, FILTER SWEEP) into the little-endian file the app reads (Resources/moving.bin).

  python export_moving_model.py checkpoint.pt out.bin

The checkpoint is what the trainer (Research/moving/train.py) saves: dict(cfg=dict(cin, ch, bpool), state=...). In
training every 3x3 layer is followed by a scaling step (batch normalisation); here that step is folded into the layer's
own weights (w' = w * s, b' = (b - mean) * s + shift with s = gain / sqrt(variance + 1e-5)), so the app runs plain layers.

Layout (all little-endian): magic 'DRMV', u32 version (1), u32 input planes (4), u32 number of levels, u32 number of
levels (from the second on) that halve the bands as well as the frames, u32 channels of each level (finest first). Then
the layers, each as u32 count + f32 weights ([out][in][time][band]) and u32 count + f32 bias, in this order: the way down
(two 3x3 layers per level, finest level first), the way up (two 3x3 layers per level, starting one above the coarsest),
the 1x1 head (three outputs per cell: the curve in units of 10 dB, the scores for 'filter sweep' and 'phaser')."""
import struct, sys
import numpy as np

MAGIC, VERSION, EPS = b'DRMV', 1, 1e-5

def layers(cin, ch):
    """(name in the checkpoint, inputs, outputs, kernel size) of every layer, in file order."""
    out = []; c0 = cin
    for i, c in enumerate(ch): out += [('enc.%d.0' % i, c0, c, 3), ('enc.%d.1' % i, c, c, 3)]; c0 = c
    for j, i in enumerate(range(len(ch) - 2, -1, -1)): out += [('dec.%d.0' % j, ch[i + 1] + ch[i], ch[i], 3), ('dec.%d.1' % j, ch[i], ch[i], 3)]
    return out + [('head', ch[0], 3, 1)]

def folded(state, name):
    """One layer's weights and bias as the app runs them: the scaling step that follows the layer in training folded in.
    (A checkpoint read back from an exported file has plain layers: taken as they are.)"""
    if name + '.weight' in state: return np.asarray(state[name + '.weight'], dtype=np.float64), np.asarray(state[name + '.bias'], dtype=np.float64)
    w = np.asarray(state[name + '.0.weight'], dtype=np.float64); b = np.asarray(state[name + '.0.bias'], dtype=np.float64)
    s = np.asarray(state[name + '.1.weight'], dtype=np.float64) / np.sqrt(np.asarray(state[name + '.1.running_var'], dtype=np.float64) + EPS)
    return w * s[:, None, None, None], (b - np.asarray(state[name + '.1.running_mean'], dtype=np.float64)) * s + np.asarray(state[name + '.1.bias'], dtype=np.float64)

def export(ck, path):
    """ck: the loaded checkpoint. Writes the file and returns the number of parameters."""
    if not (isinstance(ck, dict) and 'state' in ck and 'ch' in ck.get('cfg', {})): raise ValueError('not a checkpoint of the moving-tone model: no cfg with its levels')
    cin, ch, bpool, state = int(ck['cfg'].get('cin', 4)), [int(c) for c in ck['cfg']['ch']], int(ck['cfg'].get('bpool', 3)), ck['state']
    if cin != 4: raise ValueError('the app reads models of 4 input planes, not %d' % cin)
    if not 2 <= len(ch) <= 10 or not 0 <= bpool < len(ch): raise ValueError('levels %s with %d band halvings: not a shape the app reads' % (ch, bpool))
    blob = [MAGIC, struct.pack('<IIII', VERSION, cin, len(ch), bpool), struct.pack('<%dI' % len(ch), *ch)]; total = 0
    for name, i, o, k in layers(cin, ch):
        w, b = folded(state, name)
        if w.shape != (o, i, k, k) or b.shape != (o,): raise ValueError('%s has shape %s, expected %s' % (name, w.shape, (o, i, k, k)))
        if not (np.isfinite(w).all() and np.isfinite(b).all()): raise ValueError('%s holds numbers that are not finite' % name)
        for a in (w.astype('<f4'), b.astype('<f4')): blob += [struct.pack('<I', a.size), a.tobytes()]; total += a.size
    with open(path, 'wb') as f: f.write(b''.join(blob))
    return total

def read(path):
    """An exported file back as dict(cfg=dict(cin, ch, bpool), state=...) with plain layers ('enc.0.0.weight' ...) as float32
    arrays: for checks of the app's side against the trainer's when only the exported file is at hand."""
    blob = open(path, 'rb').read()
    if blob[:4] != MAGIC or len(blob) < 20: raise ValueError('not a moving-tone model file')
    version, cin, levels, bpool = struct.unpack('<IIII', blob[4:20])
    if version != VERSION or not 2 <= levels <= 10 or len(blob) < 20 + 4 * levels: raise ValueError('not a moving-tone model file this script knows')
    ch = list(struct.unpack('<%dI' % levels, blob[20:20 + 4 * levels])); at = 20 + 4 * levels; state = {}
    for name, i, o, k in layers(cin, ch):
        for key, shape in ((name + '.weight', (o, i, k, k)), (name + '.bias', (o,))):
            if at + 4 > len(blob): raise ValueError('the file is cut short')
            n = struct.unpack('<I', blob[at:at + 4])[0]; at += 4
            if n != int(np.prod(shape)) or at + 4 * n > len(blob): raise ValueError('%s does not have the size the header gives' % key)
            state[key] = np.frombuffer(blob, '<f4', n, at).reshape(shape).copy(); at += 4 * n
    if at != len(blob): raise ValueError('more data than the header gives')
    return dict(cfg=dict(cin=cin, ch=ch, bpool=bpool), state=state)

if __name__ == '__main__':
    if len(sys.argv) != 3: sys.exit(__doc__)
    import torch
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    ck['state'] = {k: v.numpy() for k, v in ck['state'].items()}
    n = export(ck, sys.argv[2])
    print('%s: levels %s, %d band halvings, %d parameters' % (sys.argv[2], ck['cfg']['ch'], ck['cfg'].get('bpool', 3), n))
