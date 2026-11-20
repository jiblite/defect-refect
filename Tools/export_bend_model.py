#!/usr/bin/env python3
"""Writes the bend meter (Research/bend/btrain.py: a .pt file) as the binary the app reads (Source/Core/Bend.h,
Meter::load):
    python3 Tools/export_bend_model.py Resources/bend.bin meter.pt
File: "DRBM", u32 version 1, u32 inputs per half-wave (184), u32 levels (11), u32 layers; the inputs' means and their
spreads, each as u32 count + float32; per layer u32 width, then the weights [out][in] and the biases the same way.
Little-endian."""
import struct, sys
import numpy as np, torch

def arr(f, a):
    a = np.ascontiguousarray(a, dtype='<f4').ravel(); assert np.isfinite(a).all(); f.write(struct.pack('<I', a.size)); f.write(a.tobytes())

def main(out, path):
    d = torch.load(path, map_location='cpu'); sd = d['state']; layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('f.')})
    with open(out, 'wb') as f:
        f.write(b'DRBM'); f.write(struct.pack('<IIII', 1, sd['mean'].numel(), sd['f.%d.weight' % layers[-1]].shape[0] - 1, len(layers))); arr(f, sd['mean'].numpy()); arr(f, sd['std'].numpy())
        for i in layers:
            w = sd['f.%d.weight' % i].numpy(); f.write(struct.pack('<I', w.shape[0])); arr(f, w); arr(f, sd['f.%d.bias' % i].numpy())
    print('wrote', out, 'layers', len(layers))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
