# The moving-tone model: reads the band-level picture of a sound (96 bands, 21 ms frames) and says, for every band and
# moment, by how many dB a moving filter has changed the level there, and for every moment whether a filter sweep or a
# phaser is at work. Plain 3x3 layers at six scales: the picture is halved in time five times and across the bands three
# times on the way down, and brought back up with the finer levels joined in.
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
GS = 10.0                                                    # the model's unit for the curve: 10 dB

class UNet(nn.Module):
    def __init__(s, cin=4, ch=(12, 16, 24, 32, 48, 48), bpool=3):
        super().__init__(); s.cfg = dict(cin=cin, ch=list(ch), bpool=bpool); s.enc = nn.ModuleList(); s.dec = nn.ModuleList(); c0 = cin; s.bpool = bpool
        def layer(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b))      # the scaling is folded into the layer's weights when the model is written out for the app
        for c in ch: s.enc.append(nn.ModuleList([layer(c0, c), layer(c, c)])); c0 = c
        for i in range(len(ch) - 2, -1, -1): s.dec.append(nn.ModuleList([layer(ch[i + 1] + ch[i], ch[i]), layer(ch[i], ch[i])]))
        s.head = nn.Conv2d(ch[0], 3, 1); nn.init.normal_(s.head.weight, std=0.05); nn.init.zeros_(s.head.bias); s.mult = 2 ** (len(ch) - 1)
        for m in s.modules():
            if isinstance(m, nn.Conv2d) and m is not s.head: nn.init.kaiming_normal_(m.weight, nonlinearity='relu'); nn.init.zeros_(m.bias)
    def forward(s, x):
        """x (N, cin, T, B) -> curve (N, T, B) in units of 10 dB, per-frame scores (N, T, 2) for 'filter sweep' and 'phaser'
        (against 0 for 'nothing')."""
        T = x.shape[2]; pad = (-T) % s.mult
        if pad: x = F.pad(x, (0, 0, 0, pad), mode='replicate')
        skips = []
        for i, (a, b) in enumerate(s.enc):
            if i: x = F.avg_pool2d(x, (2, 2 if i <= s.bpool else 1))
            x = torch.relu(b(torch.relu(a(x)))); skips.append(x)
        for j, (a, b) in enumerate(s.dec):
            k = len(s.enc) - 2 - j; x = x.repeat_interleave(2, dim=2)
            if k + 1 <= s.bpool: x = x.repeat_interleave(2, dim=3)
            x = torch.cat([x, skips[k]], 1); x = torch.relu(b(torch.relu(a(x))))
        z = s.head(x)[:, :, :T]; return z[:, 0], z[:, 1:].mean(3).permute(0, 2, 1)

def loss_fn(g, c, G, live, cls, wc=1.0, delta=1.5):
    """g, c: the model's answers; G (N, T, B) the true curve in dB; live (N, T, B) cells the dry sound fills; cls (N, T) 0, 1, 2.
    Curve: squared error in units of 10 dB (straight beyond 15 dB), cells the dry sound does not fill count a quarter. A plain
    absolute error will not do: more than half of all cells are untouched, so 'nothing anywhere' is its best lazy answer and the
    model never leaves it. Class: cross-entropy per frame."""
    w = 0.25 + 0.75 * live; lg = (F.huber_loss(g, G / GS, reduction='none', delta=delta) * w).sum((1, 2)) / w.sum((1, 2))
    logit = torch.cat([torch.zeros_like(c[..., :1]), c], -1); lc = F.cross_entropy(logit.reshape(-1, 3), cls.reshape(-1).long(), reduction='none').reshape(cls.shape).mean(1)
    return lg, lc, lg + wc * lc

def probs(c):
    """Per-frame scores (T, 2) -> probabilities (T, 3): nothing, filter sweep, phaser."""
    z = np.concatenate([np.zeros((len(c), 1)), c], 1); z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)
