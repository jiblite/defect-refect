# The "what room is this" model: reads the same band-level picture as the "room or sound" model and gives, per band, the
# room's strength (reverb energy per unit of direct energy, dB) and its decay time. Each band's answer is an average over
# time in which the model itself chooses the moments that count (the free decays for the decay time, and so on).
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
KMIN, KMAX = -30.0, 20.0                    # strength range in dB
TMIN, TMAX = 0.08, 14.0                     # decay time range in s
class RoomNet(nn.Module):
    def __init__(s, cin=4, ch=(16, 24, 32, 48, 64, 64)):
        super().__init__(); s.cfg = dict(cin=cin, ch=list(ch)); s.enc = nn.ModuleList(); c0 = cin
        for c in ch: s.enc.append(nn.ModuleList([nn.Conv2d(c0, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)])); c0 = c
        s.head = nn.Conv2d(ch[-1], 4, 1); s.mult = 2 ** (len(ch) - 1)
    def forward(s, x):
        T = x.shape[2]; pad = (-T) % s.mult
        if pad: x = F.pad(x, (0, 0, 0, pad), mode='replicate')
        for i, (a, b) in enumerate(s.enc):
            if i: x = F.avg_pool2d(x, (2, 1))
            x = torch.relu(b(torch.relu(a(x))))
        h = s.head(x)                                                              # (N, 4, T/32, B): two "which moments count" maps and two local answers
        wk = torch.softmax(h[:, 0], 1); wt = torch.softmax(h[:, 2], 1); k = (wk * h[:, 1]).sum(1); t = (wt * h[:, 3]).sum(1)
        return k, t                                                                # (N, B) each: strength in units of 10 dB, decay time as log2(seconds)
def targets(kdb, T60):
    """numpy (.., B) -> the two training targets and where the decay time is known."""
    return np.clip(kdb, KMIN, KMAX) / 10.0, np.log2(np.clip(np.nan_to_num(T60, nan=1.0), TMIN, TMAX)), np.isfinite(T60)
def numbers(k, t):
    """Model outputs (numpy) -> strength in dB, decay time in s."""
    return np.clip(k * 10.0, KMIN, KMAX), np.clip(2.0 ** t, TMIN, TMAX)
