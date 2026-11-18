# The moving-tone model on the scoreboard's cases (outside effects, sounds it has never seen): what it says is on, and what
# its curve does when it is taken off the wet sound and put on the case's other sound.
#   python3 ev.py model family [family ...]        (family 'clean' = the dry corpus sounds; '--all' = say-so only for every family)
import sys, os, json, glob, numpy as np, torch
sys.path[:0] = ['/tmp/claude-0/v2/moving', '/tmp/claude-0/v2/sweep']
import pic, net
from lab import cases, load, score
torch.set_num_threads(4)
def model_of(tag):
    st = torch.load('/tmp/claude-0/v2/moving/%s.pt' % tag, map_location='cpu', weights_only=False); m = net.UNet(cin=st['cfg']['cin'], ch=tuple(st['cfg']['ch']), bpool=st['cfg']['bpool']); m.load_state_dict(st['state']); m.eval(); return m
def read(m, x):
    """One channel -> (curve (T, NB) dB, probabilities (T, 3))."""
    with torch.no_grad(): g, c = m(torch.from_numpy(pic.inputs(pic.picture(x)))[None])
    return g[0].numpy().astype(np.float64) * net.GS, net.probs(c[0].numpy().astype(np.float64))
def look(m, w):
    """Stereo sound -> (curves per channel, probabilities per channel, score = the surer channel's share of frames with a moving filter, kind 1 or 2)."""
    r = [read(m, w[:, ch]) for ch in range(2)]; s = [float(1.0 - p[:, 0].mean()) for _, p in r]; k = int(np.argmax(s)); p = r[k][1]
    return [g for g, _ in r], [p for _, p in r], s[k], 1 if p[:, 1].sum() >= p[:, 2].sum() else 2
if __name__ == '__main__':
    m = model_of(sys.argv[1]); fams = sys.argv[2:]; NAME = {1: 'sweep', 2: 'phaser'}
    for fam in fams:
        if fam == 'clean':
            meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); import soundfile as sf; sc = []
            for cid in sorted(meta):
                x = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % cid, dtype='float64', always_2d=True)[0]; x = x if x.shape[1] > 1 else np.repeat(x, 2, 1); G, P, s, k = look(m, x[:, :2]); sc.append((s, cid, k))
            v = np.array([a[0] for a in sc]); print('clean corpus, %d sounds: score over 0.5 on %d, over 0.3 on %d, over 0.1 on %d; highest: %s' % (len(v), (v > 0.5).sum(), (v > 0.3).sum(), (v > 0.1).sum(), ', '.join('%s %.2f %s' % (c, s, NAME[k]) for s, c, k in sorted(sc, reverse=True)[:6])), flush=True)
            continue
        rows = []; only = fam.startswith('?'); fam = fam.lstrip('?')
        for d in cases(fam):
            cA = load(d, 'A'); cB = load(d, 'B')
            for c, o, tag in ((cA, cB, 'A'), (cB, cA, 'B')):
                G, P, s, k = look(m, c['wet'])
                if only: rows.append((s, 0, 0, k)); print('%-24s %s score %.2f %s' % (c['name'], tag, s, NAME[k]), flush=True); continue
                out = np.stack([pic.render(c['wet'][:, ch], G[ch], -1.0) for ch in range(2)], 1); onto = np.stack([pic.render(o['dry'][:, ch], G[ch], 1.0) for ch in range(2)], 1)
                r = score.removed(c['dry'], c['wet'], out)[0]; ro = score.removed(o['wet'], o['dry'], onto)[0]; rows.append((s, r, ro, k))
                print('%-20s %s score %.2f %-6s | out %7.1f onto %7.1f' % (c['name'], tag, s, NAME[k], r, ro), flush=True)
        v = np.array(rows); f = v[:, 0] > 0.5
        if only: print('%s: score over 0.5 on %d of %d sounds' % (fam, f.sum(), len(v)), flush=True)
        else: print('%s: over 0.5 on %d of %d | out, all: median %.1f mean %.1f worse %d | where over 0.5: median %.1f | onto, all: median %.1f mean %.1f worse %d | where over 0.5: median %.1f' % (fam, f.sum(), len(v), np.median(v[:, 1]), v[:, 1].mean(), (v[:, 1] < 0).sum(), np.median(v[f, 1]) if f.any() else np.nan, np.median(v[:, 2]), v[:, 2].mean(), (v[:, 2] < 0).sum(), np.median(v[f, 2]) if f.any() else np.nan), flush=True)
