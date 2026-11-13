# The "room or sound" model: reads the band-level picture of a sound (41 bands, 10.7 ms frames) and says, for every band and
# moment, how far to turn it down to be left with the direct sound (0 .. -30 dB). Small on purpose: it has to run inside the app.
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
GMIN = 30.0
class Block(nn.Module):
    """x + mix(relu(spread(relu(x)))): spread looks at each channel's 3x3 neighbours, d frames apart in time and one band apart
    (written out as nine shifted sums: fast on a CPU and trivial to repeat in C++); mix combines the channels."""
    def __init__(s, c, d):
        super().__init__(); s.d = d; s.w = nn.Parameter(torch.randn(c, 3, 3) * (1.0 / 3.0)); s.b = nn.Parameter(torch.zeros(c)); s.pw = nn.Conv2d(c, c, 1)
    def forward(s, x):
        d = s.d; T, B = x.shape[2], x.shape[3]; xp = F.pad(torch.relu(x), (1, 1, d, d)); y = s.b[None, :, None, None]
        for i in range(3):
            for j in range(3): y = y + s.w[None, :, i, j, None, None] * xp[:, :, i * d:i * d + T, j:j + B]
        return x + s.pw(torch.relu(y))
class Net(nn.Module):
    def __init__(s, cin=4, c=24, dil=(1, 2, 4, 8, 16, 32, 64, 1, 4, 16)):
        super().__init__(); s.stem = nn.Conv2d(cin, c, (3, 3), padding=1); s.blocks = nn.ModuleList([Block(c, d) for d in dil]); s.head = nn.Conv2d(c, 1, 1); s.cfg = dict(cin=cin, c=c, dil=list(dil))
        nn.init.zeros_(s.head.weight); nn.init.constant_(s.head.bias, -2.0)
        for b in s.blocks: nn.init.normal_(b.pw.weight, std=0.3 / np.sqrt(c))
    def forward(s, x):
        x = s.stem(x)
        for b in s.blocks: x = b(x)
        z = s.head(torch.relu(x))[:, 0]; return (-GMIN * torch.sigmoid(z), z) if s.training else -GMIN * torch.sigmoid(z)            # (N, T, B) gain in dB
def fast_drops(LE, k, lo=45.0, tail=6):
    """Per band: how fast the level can fall in this sound: the 95th percentile, over the whole sound, of the fall in dB
    within k frames, counted from cells no more than `lo` dB under the loudest cell (the last frames, which run into the
    silence behind the sound, are left out). A sound in a room never falls faster than the room lets it; a dry sound stops
    dead. Bands with fewer than 20 such cells get the median of the others (0 if there is none)."""
    T, B = LE.shape; n = T - tail - k; out = np.full(B, np.nan)
    if n > 0:
        d = LE[:n] - LE[k:k + n]; ok = LE[:n] > LE.max() - lo
        for b in range(B):
            if ok[:, b].sum() >= 20: out[b] = np.percentile(d[ok[:, b], b], 95)
    return np.where(np.isfinite(out), out, np.nanmedian(out) if np.isfinite(out).any() else 0.0)
def inputs(LE, U64, U256, drops=False):
    """Arrays (T, B) in dB -> model input (4, T, B); with drops two more planes that hold one number per band for the whole
    sound: the fastest falls within 43 ms and within 171 ms (fast_drops), 30 and 40 dB = 1."""
    top = LE.max(); e = np.clip((LE - top) / 20.0, -3.5, 0.0) + 1.75; u1 = np.clip(U64, -40.0, 0.0) / 20.0 + 1.0; u2 = np.clip(U256, -40.0, 0.0) / 20.0 + 1.0
    pos = np.broadcast_to(np.linspace(-1.0, 1.0, LE.shape[1])[None, :], LE.shape); pl = [e, u1, u2, pos]
    if drops:
        LEd = LE.astype(np.float64); pl += [np.broadcast_to(np.clip(fast_drops(LEd, 4) / 30.0, 0.0, 1.2)[None, :], LE.shape), np.broadcast_to(np.clip(fast_drops(LEd, 16) / 40.0, 0.0, 1.2)[None, :], LE.shape)]
    return np.stack(pl).astype(np.float32)
def loss_fn(g, LE, LS, top):
    """The scoreboard's distance, at band level: |level after the gain - level of the true direct part| in dB, with the floor
    60 dB under the sound's loudest band and a cap of 30 dB per cell."""
    fl = (top - 60.0)[:, None, None]; out = torch.maximum(LE + g, fl); ref = torch.maximum(LS, fl); return torch.clamp((out - ref).abs(), max=30.0)
def proxy(g, LE, LS):
    """numpy, one sound: 100 = the gains give exactly the direct part, 0 = no better than leaving the sound as it is."""
    top = LE.max(); fl = top - 60.0; ref = np.maximum(LS, fl); a = np.minimum(np.abs(np.maximum(LE + g, fl) - ref), 30.0).sum(); b = np.minimum(np.abs(np.maximum(LE, fl) - ref), 30.0).sum()
    return 100.0 * (1.0 - a / max(b, 1e-9)), b / LE.size

class UNet(nn.Module):
    """The same job with plain 3x3 layers at six time scales (10.7 ms .. 341 ms per step): the picture is halved in time on the
    way down and brought back up with the finer levels joined in. Long tails are seen at the coarse levels, onsets at the fine.
    glob > 0 adds a look at the whole sound: what the coarsest level holds is averaged over time per band (audible cells
    only), a small layer over the bands turns that into (1) an addition to the coarsest level at every moment and (2) a
    shift of the answer per band. So the model can settle how reverberant the sound is as a whole before it decides cell by
    cell (the local layers see only about 1.3 s each way). Both start at zero: the model begins as the one without."""
    def __init__(s, cin=4, ch=(12, 16, 24, 32, 48, 48), glob=0):
        super().__init__(); s.cfg = dict(cin=cin, ch=list(ch), glob=glob); s.enc = nn.ModuleList(); s.dec = nn.ModuleList(); c0 = cin; s.glob = glob
        for c in ch: s.enc.append(nn.ModuleList([nn.Conv2d(c0, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)])); c0 = c
        for i in range(len(ch) - 2, -1, -1): s.dec.append(nn.ModuleList([nn.Conv2d(ch[i + 1] + ch[i], ch[i], 3, padding=1), nn.Conv2d(ch[i], ch[i], 3, padding=1)]))
        s.head = nn.Conv2d(ch[0], 1, 1); nn.init.constant_(s.head.bias, -1.5); s.mult = 2 ** (len(ch) - 1)
        if glob:
            s.ga = nn.Conv1d(2 * ch[-1], glob, 3, padding=1); s.gb = nn.Conv1d(glob, ch[-1], 3, padding=1); s.gc = nn.Conv1d(glob, 1, 3, padding=1)
            for m in (s.gb, s.gc): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def whole(s, x, live):
        """x: the coarsest level (N, C, T', B); live (N, T', B): the share of audible cells behind each of its cells.
        Returns the per-band summary h (N, glob, B)."""
        w = live[:, None]; m = (x * w).sum(2) / w.sum(2).clamp(min=1e-3); mg = (x * w).sum((2, 3)) / w.sum((2, 3)).clamp(min=1e-3)       # per band over time; over everything
        return torch.relu(s.ga(torch.cat([m, mg[:, :, None].expand(-1, -1, m.shape[2])], 1)))
    def forward(s, x):
        T = x.shape[2]; pad = (-T) % s.mult; live = (x[:, 0] > -1.25).float() if s.glob else None            # plane 0 above -1.25 = within 60 dB of the loudest cell
        if pad:
            x = F.pad(x, (0, 0, 0, pad), mode='replicate')
            if s.glob: live = F.pad(live, (0, 0, 0, pad))                                                      # the added frames do not count
        skips = []
        for i, (a, b) in enumerate(s.enc):
            if i: x = F.avg_pool2d(x, (2, 1))
            x = torch.relu(b(torch.relu(a(x)))); skips.append(x)
        zoff = 0.0
        if s.glob:
            h = s.whole(x, F.avg_pool2d(live[:, None], (s.mult, 1))[:, 0]); x = x + s.gb(h)[:, :, None, :]; zoff = s.gc(h)                # (N, 1, B)
        for j, (a, b) in enumerate(s.dec):
            k = len(s.enc) - 2 - j; x = torch.cat([x.repeat_interleave(2, dim=2), skips[k]], 1); x = torch.relu(b(torch.relu(a(x))))
        z = s.head(x)[:, 0, :T] + zoff; return (-GMIN * torch.sigmoid(z), z) if s.training else -GMIN * torch.sigmoid(z)

def train_loss(g, z, LE, LS, top, bce_w=3.0):
    """The scoreboard distance (see loss_fn) plus a cross-entropy on 'how far down, as a share of 30 dB': the second never
    goes flat when the model sits at 'do nothing', which the first alone does. Per cell; cells under the floor count zero."""
    fl = (top - 60.0)[:, None, None]; live = (LE > fl).float(); t = torch.clamp((LE - torch.maximum(LS, fl)) / GMIN, 0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(z, t, reduction='none') - F.binary_cross_entropy(t, t, reduction='none')          # zero at the right answer
    return loss_fn(g, LE, LS, top) / 10.0 + bce_w * bce * live

def total_loss(g, LE, LS, top):
    """Per sound: how far the band totals are off, in units of 10 dB: the direct part's energy per band and the reverb
    part's energy per band (as the gains split them) against the true split. The per-cell loss asks for the right level
    in the typical cell; this asks for the right sums, which is what the room's strength is read from."""
    fl = (top - 60.0)[:, None, None]; live = (LE > fl).double(); E = 10 ** (LE.double() / 10.0) * live; St = torch.minimum(10 ** (torch.maximum(LS, fl).double() / 10.0), E) * live; Sh = E * 10 ** (g.double() / 10.0)
    es = E.sum(1); ok = (es > 1e-4 * es.max(1, keepdim=True)[0]).double(); eps = 1e-3 * es + 1e-30
    ds = (10 * torch.log10((Sh.sum(1) + eps) / (St.sum(1) + eps))).abs(); dr = (10 * torch.log10(((E - Sh).sum(1) + eps) / ((E - St).sum(1) + eps))).abs()
    return (((ds + dr) * ok).sum(1) / ok.sum(1).clamp(min=1.0) / 10.0).float()
