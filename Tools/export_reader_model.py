#!/usr/bin/env python3
"""Writes a curve reader's nets (Research/filter/ftrain.py: one .pt file per net) as the binary the app reads
(Source/Core/Reader.h, Reader::load):
    python3 Tools/export_reader_model.py Resources/filter.bin net1.pt [net2.pt ...]
File: "DRCR", u32 version 2, u32 number of nets; per net: u32 inputs (curves + 1), u32 bands, u32 layers, u32 channels,
u32 width; per layer u32 spacing, then the weights [out][in][width] and the biases, each as u32 count + float32;
then the curve head (channels -> 1) and the "there" head (2 x channels -> 1) the same way. Little-endian."""
import struct, sys
import numpy as np, torch

def arr(f, a):
    a = np.ascontiguousarray(a, dtype='<f4').ravel(); assert np.isfinite(a).all(); f.write(struct.pack('<I', a.size)); f.write(a.tobytes())

def main(out, nets):
    with open(out, 'wb') as f:
        f.write(b'DRCR'); f.write(struct.pack('<II', 2, len(nets)))
        for path in nets:
            sd = torch.load(path, map_location='cpu'); layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('c.')})
            w0 = sd['c.0.weight'].numpy(); ch, nin, width = w0.shape; dil = [1, 2, 4, 8, 1][:len(layers)]; assert len(layers) == 5, 'the reader has five layers'
            f.write(struct.pack('<IIIII', nin, 104, len(layers), ch, width))
            for i, d in zip(layers, dil):
                w = sd['c.%d.weight' % i].numpy(); assert w.shape[0] == ch and w.shape[2] == width; f.write(struct.pack('<I', d)); arr(f, w); arr(f, sd['c.%d.bias' % i].numpy())
            arr(f, sd['curve.weight'].numpy()); arr(f, sd['curve.bias'].numpy()); arr(f, sd['there.weight'].numpy()); arr(f, sd['there.bias'].numpy())
    print('wrote', out, 'nets', len(nets))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
