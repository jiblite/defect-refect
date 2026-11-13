# Builds the wet/dry test sets: every zoo effect on dry clips, twice (sound A and sound B through the same settings), plus the
# effect's response to a click on the left channel only and on the right channel only (the full 2x2 response).
import numpy as np, soundfile as sf, json, os, sys, hashlib, scipy.signal as ss
sys.path.insert(0, '/tmp/claude-0/v2/bench')
from zoo import *
from multiprocessing import Pool
CORPUS = '/tmp/claude-0/v2/corpus/'; OUT = '/tmp/claude-0/v2/bench/sets/'; CLICK = 480
meta = json.load(open(CORPUS + 'meta.json'))
def dry(name):
    x, sr = sf.read(CORPUS + 'dry/' + name + '.wav', always_2d=True); return x.astype(np.float64)
def wflac(path, x, g=None):
    pk = np.abs(x).max()
    if g is None: g = 1.0 if pk < 0.98 else 0.98 / pk
    sf.write(path, x * g, SR, format='FLAC', subtype='PCM_24'); return g
def db(a, b): return float(10 * np.log10((a ** 2).sum() / max((b ** 2).sum(), 1e-30) + 1e-30))
def one(job):
    name, i, family, split, p, a, b, tail = job; d = '%s%s/%s_%02d/' % (OUT, family, name, i)
    if os.path.exists(d + 'meta.json'): return name, i, 'kept'
    os.makedirs(d, exist_ok=True)
    try:
        xa, xb = dry(a), dry(b); wa = apply(name, xa, p, tail); wb = apply(name, xb, p, tail)
        cl = impulse(SR * 6, CLICK); cl[:, 1] = 0; hl = apply(name, cl, p); cr = impulse(SR * 6, CLICK); cr[:, 0] = 0; hr = apply(name, cr, p)
    except Exception as ex:
        return name, i, 'FAILED ' + str(ex)[:200]
    if not (np.isfinite(wa).all() and np.isfinite(wb).all() and np.isfinite(hl).all() and np.isfinite(hr).all()): return name, i, 'NAN'
    # how well the click responses describe what the effect did to sound A (exact for a fixed linear effect; not for a modulated one)
    N = len(wa); pred = np.stack([ss.fftconvolve(xa[:, 0], hl[CLICK:, c])[:N] + ss.fftconvolve(xa[:, 1], hr[CLICK:, c])[:N] for c in range(2)], 1)
    ga = wflac(d + 'A_wet.flac', wa); gb = wflac(d + 'B_wet.flac', wb); pk = max(np.abs(hl).max(), np.abs(hr).max()); gi = 1.0 if pk < 0.98 else 0.98 / pk
    wflac(d + 'ir_L.flac', hl, gi); wflac(d + 'ir_R.flac', hr, gi)
    json.dump(dict(effect=name, family=family, split=split, params=p, A=a, B=b, tail=tail, pre=PRE, click=CLICK, gainA=ga, gainB=gb, gainIR=gi, fixed_db=db(wa - pred, wa)), open(d + 'meta.json', 'w'), indent=1)
    return name, i, 'built'
def jobs(family, per=8, tail=3.0, seed=7):
    names = { s: sorted(n for n, m in meta.items() if m['split'] == s) for s in ('dev', 'test') }; out = []
    for name, (fam, split, sampler, fn) in ZOO.items():
        if fam != family: continue
        rng = np.random.default_rng(int(hashlib.md5((name + str(seed)).encode()).hexdigest(), 16) % 2**32)
        for i in range(per):
            p = sampler(rng); a, b = rng.choice(names[split], 2, replace=False); out.append((name, i, family, split, p, str(a), str(b), tail))
    return out
if __name__ == '__main__':
    for fam in sys.argv[1:]:
        with Pool(4) as pool:
            for k, r in enumerate(pool.imap_unordered(one, jobs(fam))):
                if r[2] != 'built' or k % 16 == 0: print(k, r, flush=True)
