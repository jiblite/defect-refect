# New test sounds for the REVERB sureness rule, none of them the scoreboard's own (other settings, other pairings):
#   fx   every tool of the other 17 effect families on a clean corpus clip (6 s, cut off, as the scoreboard's)      -> no reverb
#   rev  every reverb tool on a clean clip, ringing out (9 s); looked at whole and cut off at 6 s                   -> reverb
#   fxr  an effect, then a reverb after it (9 s); looked at whole and cut off at 6 s                                -> reverb
# 'dev' tools get 'dev' clips, 'test' tools 'test' clips, as on the scoreboard. WAV files (16 bit) in roomsure/wav/nw.
#   python3 prep_nw.py [processes]     -> nw_rows.json, nw_list.txt
import sys, os, json, hashlib
sys.dont_write_bytecode = True; sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/detect/py')
import numpy as np, soundfile as sf
import zoo as R, zoo_fx as Z
from multiprocessing import Pool
SR = 48000; CORPUS = '/tmp/claude-0/v2/corpus/'; OUT = '/tmp/claude-0/v2/roomsure/wav/nw/'; os.makedirs(OUT, exist_ok=True); meta = json.load(open(CORPUS + 'meta.json'))
NAMES = {s: sorted(n for n, m in meta.items() if m['split'] == s) for s in ('dev', 'test')}
REVS = {s: sorted(n for n, v in R.ZOO.items() if v[0] == 'reverb' and v[1] == s) for s in ('dev', 'test')}
def dry(n): return sf.read(CORPUS + 'dry/' + n + '.wav', always_2d=True)[0].astype(np.float64)
def db(a, b): return float(10 * np.log10((a ** 2).sum() / max((b ** 2).sum(), 1e-30) + 1e-30))
def change_db(x, w):
    n = len(x); g = (w[:n] * x).sum() / ((x ** 2).sum() + 1e-30); return db(w[:n] - g * x, x * g) if abs(g) > 1e-3 else 99.0
def rng_of(*a): return np.random.default_rng(int(hashlib.md5(('nw/' + '/'.join(str(v) for v in a)).encode()).hexdigest(), 16) % 2**32)
def save(path, y):
    pk = np.abs(y).max(); sf.write(path, y * (0.98 / pk if pk >= 0.98 else 1.0), SR, subtype='PCM_16')
def fx_draw(name, rng, tail):
    fam, split, samp, fn, clipf = Z.ZOO[name]
    for tries in range(8):
        p = samp(rng); a = str(rng.choice(NAMES[split])); x = dry(a)
        if clipf is not None and not clipf(x): continue
        w = Z.apply(name, x, p, tail)
        if np.isfinite(w).all() and change_db(x, w) > -30 and np.abs(w).max() > 1e-3: return a, p, w
    return None
def one(job):
    kind, name, i = job; path = OUT + '%s__%s__%02d.wav' % (kind, name, i); rng = rng_of(kind, name, i)
    try:
        if kind == 'fx':
            r = fx_draw(name, rng, 0.0)
            if r is None: return None
            a, p, w = r; row = dict(fam=Z.ZOO[name][0], tool=name, split=Z.ZOO[name][1], dry=a, wet=0)
        elif kind == 'rev':
            fam, split, samp, fn = R.ZOO[name][:4]; p = samp(rng); a = str(rng.choice(NAMES[split])); w = R.apply(name, dry(a), p, 3.0); row = dict(fam='reverb', tool=name, split=split, dry=a, wet=1)
        else:
            r = fx_draw(name, rng, 3.0)
            if r is None: return None
            a, p, w = r; split = Z.ZOO[name][1]; rv = str(rng.choice(REVS[split])); w = R.apply(rv, w * (0.5 / max(np.abs(w).max(), 1e-9)), R.ZOO[rv][2](rng), 0.0); row = dict(fam=Z.ZOO[name][0], tool=name, split=split, dry=a, wet=1, reverb=rv)
        if not np.isfinite(w).all(): return None
        if not os.path.exists(path): save(path, w)
        return dict(row, path=path, set='nw', kind=kind, case='%s_%02d' % (name, i), sec=len(w) / SR)
    except Exception as e:
        print('FAILED', job, str(e)[:160], flush=True); return None
if __name__ == '__main__':
    jobs = [('fx', n, i) for n in Z.ZOO for i in range(10)] + [('rev', n, i) for n, v in R.ZOO.items() if v[0] == 'reverb' for i in range(16)] + [('fxr', n, i) for n in Z.ZOO for i in range(4)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as p: rows = [r for r in p.map(one, jobs, chunksize=4) if r is not None]
    out = []
    for r in rows:
        out.append(dict(r, cut=0))
        if r['kind'] != 'fx': out.append(dict(r, cut=6))          # the ringing sounds are looked at a second time, cut off at 6 s
    json.dump(out, open('/tmp/claude-0/v2/roomsure/nw_rows.json', 'w')); open('/tmp/claude-0/v2/roomsure/nw_list.txt', 'w').write(''.join(r['path'] + ('\t6' if r['cut'] else '') + '\n' for r in out))
    import collections; print(len(jobs), 'jobs,', len(rows), 'sounds,', len(out), 'looks', dict(collections.Counter(r['kind'] for r in rows)))
