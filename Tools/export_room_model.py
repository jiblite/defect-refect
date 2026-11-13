"""Export the trained room model (the small network that says, per band and moment, how far a sound has to come
down to leave the direct sound) into the little-endian file the app reads (e.g. Resources/room.bin).

  python export_room_model.py checkpoint.pt out.bin [deepest_cut_db]

The checkpoint is what the trainer saves: dict(cfg=dict(cin, ch), state=...). deepest_cut_db is the trainer's
output range (the answer runs from 0 down to minus this; 30 unless the trainer was changed).

Layout (all little-endian): magic 'DRRM', u32 version (1), u32 input planes (4, or 7 for a model that runs as a
second pass), f32 deepest cut in dB, u32 number of levels, u32 channels of each level (finest first). Then the
layers, each as u32 count + f32 weights ([out][in][time][band], as trained) and u32 count + f32 bias, in this
order: the way down (two 3x3 layers per level, finest level first), the way up (two 3x3 layers per level, starting
one above the coarsest), the 1x1 head."""
import struct, sys
import numpy as np

MAGIC, VERSION = b'DRRM', 1

def layers(cin, ch):
    """(name in the checkpoint, inputs, outputs, kernel size) of every layer, in file order."""
    out = []; c0 = cin
    for i, c in enumerate(ch): out += [('enc.%d.0' % i, c0, c, 3), ('enc.%d.1' % i, c, c, 3)]; c0 = c
    for j, i in enumerate(range(len(ch) - 2, -1, -1)): out += [('dec.%d.0' % j, ch[i + 1] + ch[i], ch[i], 3), ('dec.%d.1' % j, ch[i], ch[i], 3)]
    return out + [('head', ch[0], 1, 1)]

def export(ck, path, deepest=30.0):
    """ck: the loaded checkpoint. Writes the file and returns the number of parameters."""
    if not (isinstance(ck, dict) and 'state' in ck and 'ch' in ck.get('cfg', {})): raise ValueError('not a checkpoint of the room model: no cfg with its levels')
    if ck['cfg'].get('glob', 0): raise ValueError('this model takes a look at the whole sound (glob), which the app does not have')
    cin, ch, state = int(ck['cfg'].get('cin', 4)), [int(c) for c in ck['cfg']['ch']], ck['state']
    if cin not in (4, 7): raise ValueError('the app reads models of 4 or 7 input planes, not %d' % cin)
    want = layers(cin, ch); names = {n + s for n, _, _, _ in want for s in ('.weight', '.bias')}
    if set(state.keys()) != names: raise ValueError('the checkpoint does not hold the layers its sizes give: ' + ', '.join(sorted(set(state.keys()) ^ names)))
    blob = [MAGIC, struct.pack('<IIfI', VERSION, cin, deepest, len(ch)), struct.pack('<%dI' % len(ch), *ch)]; total = 0
    for name, i, o, k in want:
        w = np.asarray(state[name + '.weight'], dtype='<f4'); b = np.asarray(state[name + '.bias'], dtype='<f4')
        if w.shape != (o, i, k, k) or b.shape != (o,): raise ValueError('%s has shape %s, expected %s' % (name, w.shape, (o, i, k, k)))
        if not (np.isfinite(w).all() and np.isfinite(b).all()): raise ValueError('%s holds numbers that are not finite' % name)
        for a in (w, b): blob += [struct.pack('<I', a.size), a.tobytes()]; total += a.size
    with open(path, 'wb') as f: f.write(b''.join(blob))
    return total

def read(path):
    """An exported file back as a checkpoint, dict(cfg=dict(cin, ch), state=..., deepest=...), the layers as float32
    arrays: for checks of the app's side against the trainer's when only the exported file is at hand."""
    blob = open(path, 'rb').read()
    if blob[:4] != MAGIC or len(blob) < 20: raise ValueError('not a room model file')
    version, cin, deepest, levels = struct.unpack('<IIfI', blob[4:20])
    if version != VERSION or not 1 <= levels <= 12 or len(blob) < 20 + 4 * levels: raise ValueError('not a room model file this script knows')
    ch = list(struct.unpack('<%dI' % levels, blob[20:20 + 4 * levels])); at = 20 + 4 * levels; state = {}
    for name, i, o, k in layers(cin, ch):
        for key, shape in ((name + '.weight', (o, i, k, k)), (name + '.bias', (o,))):
            if at + 4 > len(blob): raise ValueError('the file is cut short')
            n = struct.unpack('<I', blob[at:at + 4])[0]; at += 4
            if n != int(np.prod(shape)) or at + 4 * n > len(blob): raise ValueError('%s does not have the size the header gives' % key)
            state[key] = np.frombuffer(blob, '<f4', n, at).reshape(shape).copy(); at += 4 * n
    if at != len(blob): raise ValueError('more data than the header gives')
    return dict(cfg=dict(cin=cin, ch=ch), state=state, deepest=deepest)

if __name__ == '__main__':
    if len(sys.argv) not in (3, 4): sys.exit(__doc__)
    import torch
    ck = torch.load(sys.argv[1], map_location='cpu')
    n = export(ck, sys.argv[2], float(sys.argv[3]) if len(sys.argv) == 4 else 30.0)
    print('%s: levels %s, %d parameters' % (sys.argv[2], ck['cfg']['ch'], n))
