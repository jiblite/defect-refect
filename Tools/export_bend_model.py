#!/usr/bin/env python3
"""Writes the bend meter (Research/bend/btrain.py: one .pt file per net) as the binary the app reads
(Source/Core/Bend.h, Meter::load):
    python3 Tools/export_bend_model.py Resources/bend.bin meter_a.pt [meter_b.pt ...]
File: "DRBM", u32 version 2, u32 inputs per half-wave (184), u32 levels (11), u32 nets; per net: u32 layers, the
inputs' means and their spreads, each as u32 count + float32, and per layer u32 width, then the weights [out][in]
and the biases the same way. Little-endian."""
import struct, sys
import numpy as np, torch

def arr(f, a):
    a = np.ascontiguousarray(a, dtype='<f4').ravel(); assert np.isfinite(a).all(); f.write(struct.pack('<I', a.size)); f.write(a.tobytes())

def main(out, paths):
    nets = [torch.load(p, map_location='cpu')['state'] for p in paths]
    with open(out, 'wb') as f:
        f.write(b'DRBM'); f.write(struct.pack('<IIII', 2, 184, 11, len(nets)))
        for sd in nets:
            layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('f.')}); assert sd['mean'].numel() == 184 and sd['f.%d.weight' % layers[-1]].shape[0] == 12
            f.write(struct.pack('<I', len(layers))); arr(f, sd['mean'].numpy()); arr(f, sd['std'].numpy())
            for i in layers:
                w = sd['f.%d.weight' % i].numpy(); f.write(struct.pack('<I', w.shape[0])); arr(f, w); arr(f, sd['f.%d.bias' % i].numpy())
    print('wrote', out, 'nets', len(nets))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
