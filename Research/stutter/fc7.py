# Fresh cases for the crush and the stutter engines: n draws for each outside effect of the family (zoo_fx.py) and for
# plain versions of my own, each on two clips of the corpus (A: what the engine reads; B: what its answer is put onto,
# to be compared with B through the same effect at the same settings).
#   python3 fc7.py make bitcrush|stutter n    -> gen_<family>/<effect>_<i>_{A,B}.flac, gen_<family>/meta.json
import sys, os, json, hashlib
sys.path.insert(0, '/tmp/claude-0/v2/newfx'); sys.path.insert(0, '/home/claude/defect-refect/Research/detect')
from st_common import *
def plain_crush(x, p):
    """Rounded to a grid (step, offset) and held: boundaries ceil(j P + c)."""
    q = p['offset'] + p['step'] * np.round((x - p['offset']) / p['step']); n = len(x)
    if p['P'] <= 1.0: return q
    j = np.arange(-2, int(n / p['P']) + 3); b = np.unique(np.clip(np.ceil(j * p['P'] + p['c'] - 1e-9).astype(np.int64), 0, n - 1)); src = np.zeros(n, np.int64); src[b] = b; src = np.maximum.accumulate(src); return q[src]
def crush_draw(r):
    bits = float(r.uniform(3.0, 10.0)) if r.random() < 0.5 else float(r.integers(3, 11)); step = 2.0 ** (1.0 - bits); kind = r.integers(0, 3)
    P = 1.0 if kind == 0 else (float(r.integers(2, 13)) if kind == 1 else float(r.uniform(1.5, 12.0)))
    return dict(bits=bits, step=step, offset=float(0.0 if r.random() < 0.5 else 0.5 * step), P=P, c=float(r.uniform(0, P)) if P > 1 else 0.0)
def plain_stutter(x, p):
    o = x.copy(); n = len(x)
    for s0, L, end in p['events']:
        if s0 + L > n: continue
        sl = x[s0:s0 + L].copy()
        if p['fades'] and L >= 192: rr = (np.arange(96) / 96)[:, None]; sl[:96] *= rr; sl[L - 96:] *= rr[::-1]
        e = min(end, n); k = e - s0; o[s0:e] = np.tile(sl, (k // L + 1, 1))[:k]
    return o
def stutter_draw(r, n=288000):
    ev = []; pos = int(r.uniform(0, 0.5) * SR); same = r.random() < 0.5; L0 = int(np.exp(r.uniform(np.log(0.025), np.log(0.4))) * SR)
    while pos < n:
        L = L0 if same else int(np.exp(r.uniform(np.log(0.025), np.log(0.4))) * SR); reps = int(r.integers(2, 9)); end = pos + reps * L
        if r.random() < 0.2: end -= int(r.uniform(0.2, 0.8) * L)                       # the last play cut short
        if end > n: break
        ev.append((pos, L, end)); pos = end + (0 if r.random() < 0.3 else int(r.uniform(0.05, 1.0) * SR))
    return dict(events=ev, fades=bool(r.random() < 0.6))
if __name__ == '__main__' and sys.argv[1] == 'make':
    import zoo_fx as Z
    fam = sys.argv[2]; G = '/tmp/claude-0/v2/newfx/gen_%s/' % fam; os.makedirs(G, exist_ok=True)
    cm = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); clips = sorted(cm); meta = {}
    names = dict(bitcrush=['bit_calf', 'bit_pedalboard', 'bit_swh', 'bit_mda', 'plain'], stutter=['stu_sox', 'stu_ffmpeg', 'plain'])[fam]
    for name in names:
        if name == 'plain': sampler, clipf = (crush_draw if fam == 'bitcrush' else stutter_draw), None
        else: _, split, sampler, fn, clipf = Z.ZOO[name]
        for i in range(int(sys.argv[3])):
            rng = np.random.default_rng(int(hashlib.md5((name + 'gen_%s/%d' % (fam, i)).encode()).hexdigest(), 16) % 2 ** 32)
            for tries in range(8):
                p = sampler(rng); a, b = [str(v) for v in rng.choice(clips, 2, replace=False)]; xa = truth.rd('/tmp/claude-0/v2/corpus/dry/' + a + '.wav'); xb = truth.rd('/tmp/claude-0/v2/corpus/dry/' + b + '.wav')
                if clipf is not None and not (clipf(xa) and clipf(xb)): continue
                try:
                    if name == 'plain': wa, wb = ((plain_crush(xa, p), plain_crush(xb, p)) if fam == 'bitcrush' else (plain_stutter(xa, p), plain_stutter(xb, p)))
                    else: pre = 0.0 if fam == 'stutter' else Z.PRE; wa, wb = Z.apply(name, xa, p, 0.0, pre), Z.apply(name, xb, p, 0.0, pre)      # (the stutter tools go by the sample count: no silence in front, or their plan is shifted)
                except Exception as e: print(name, i, 'failed', str(e)[:100]); continue
                if np.isfinite(wa).all() and np.isfinite(wb).all() and np.abs(wa).max() > 1e-3: break
            else: print(name, i, 'no case'); continue
            g = min(1.0, 0.98 / max(np.abs(wa).max(), np.abs(wb).max()))
            for w, t in ((wa, 'A'), (wb, 'B')): sf.write(G + '%s_%02d_%s.flac' % (name, i, t), w * g, SR, format='FLAC', subtype='PCM_24')
            meta['%s_%02d' % (name, i)] = dict(effect=name, family=fam, A=a, B=b, params=p, gain=g); print(name, i, flush=True)
    json.dump(meta, open(G + 'meta.json', 'w'), indent=1)
