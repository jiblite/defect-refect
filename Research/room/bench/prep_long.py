# Long test sounds for the REVERB sureness rule (the rule was fitted on sounds of 6 and 9 s; most files are longer).
#   loopc  a clean corpus clip played six times over (36 s): its verdict should be the clip's own                      -> no reverb
#   loopr  a reverb tool on such a loop, ringing out (39 s); also looked at cut off at 18 s and at 35 s                -> reverb
#   catc   the 12 long clean sounds of Research/moving/mklong.py (five to eight different clips in a row, 30 to 48 s)  -> no reverb
#   catfx  the long sounds with a filter sweep, a phaser or a chorus of mklong.py (both folders)                       -> no reverb
#   catr   a reverb tool on each of the 12 long clean sounds plus three more draws, ringing out; also cut off at the sound's end -> reverb
# 'dev' tools get 'dev' clips and 'test' tools 'test' clips, as in prep_nw.py. WAV files (16 bit) in roomsure/wav/long.
#   python3 prep_long.py [processes]     -> long_rows.json, long_list.txt (a tab and a number after a path: only the first so many seconds)
import sys, os, json, glob, hashlib
sys.dont_write_bytecode = True; sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/detect/py')
import numpy as np, soundfile as sf
import zoo as R
from multiprocessing import Pool
SR = 48000; CORPUS = '/tmp/claude-0/v2/corpus/'; OUT = '/tmp/claude-0/v2/roomsure/wav/long/'; os.makedirs(OUT, exist_ok=True); meta = json.load(open(CORPUS + 'meta.json'))
NAMES = {s: sorted(n for n, m in meta.items() if m['split'] == s) for s in ('dev', 'test')}
REVS = {s: sorted(n for n, v in R.ZOO.items() if v[0] == 'reverb' and v[1] == s) for s in ('dev', 'test')}
MVL, CHL = '/dev/shm/mvl/', '/dev/shm/chl/'; TIMES = 6; NLOOP = 20
def dry(n): return sf.read(CORPUS + 'dry/' + n + '.wav', always_2d=True)[0].astype(np.float64)
def rng_of(*a): return np.random.default_rng(int(hashlib.md5(('long/' + '/'.join(str(v) for v in a)).encode()).hexdigest(), 16) % 2**32)
def save(path, y):
    pk = np.abs(y).max(); sf.write(path, y * (0.98 / pk if pk >= 0.98 else 1.0), SR, subtype='PCM_16')
def one(job):
    kind = job[0]
    try:
        if kind == 'loopc':
            _, split, k = job; a = NAMES[split][k * (len(NAMES[split]) // NLOOP)]; path = OUT + 'loopc__%s_%02d.wav' % (split, k); w = np.tile(dry(a), (TIMES, 1)); row = dict(split=split, dry=a, wet=0, tool='none')
        elif kind == 'loopr':
            _, name, i = job; split = R.ZOO[name][1]; rng = rng_of(kind, name, i); a = str(rng.choice(NAMES[split])); path = OUT + 'loopr__%s__%02d.wav' % (name, i)
            w = R.apply(name, np.tile(dry(a), (TIMES, 1)), R.ZOO[name][2](rng), 3.0); row = dict(split=split, dry=a, wet=1, tool=name)
        else:
            _, name, i, src = job; rng = rng_of(kind, name, i); path = OUT + 'catr__%s__%02d.wav' % (name, i); x = sf.read(MVL + src + '.wav', always_2d=True)[0].astype(np.float64)
            w = R.apply(name, x, R.ZOO[name][2](rng), 3.0); row = dict(split=R.ZOO[name][1], dry=src, wet=1, tool=name, dry_sec=len(x) / SR)
        if not np.isfinite(w).all(): return None
        if not os.path.exists(path): save(path, w)
        return dict(row, path=path, set='long', kind=kind, sec=len(w) / SR)
    except Exception as e:
        print('FAILED', job, str(e)[:160], flush=True); return None
if __name__ == '__main__':
    revs = sorted(n for n, v in R.ZOO.items() if v[0] == 'reverb'); cleans = ['clean%02d' % i for i in range(12)]
    jobs = [('loopc', s, k) for s in ('dev', 'test') for k in range(NLOOP)] + [('loopr', n, i) for n in revs for i in range(2 if R.ZOO[n][1] == 'dev' else 4)] + [('catr', revs[i % len(revs)], i, cleans[i % 12]) for i in range(15)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as p: rows = [r for r in p.map(one, jobs, chunksize=2) if r is not None]
    L = {r['name']: r for r in json.load(open(MVL + 'list.json'))}
    rows += [dict(path=MVL + n + '.wav', set='long', kind='catc', split='mixed', dry=n, wet=0, tool='none', sec=L[n]['sec']) for n in cleans]
    rows += [dict(path=p, set='long', kind='catfx', split='mixed', dry=L.get(os.path.basename(p)[:-8], {}).get('dry', ''), wet=0, tool=L.get(os.path.basename(p)[:-8], {}).get('tool', 'chorus'), sec=sf.info(p).frames / SR) for p in sorted(glob.glob(MVL + '*_wet.wav')) + sorted(glob.glob(CHL + '*_wet.wav'))]
    out = []
    for r in rows:
        out.append(dict(r, cut=0))
        if r['kind'] == 'loopc': out.append(dict(r, cut=18))
        if r['kind'] == 'loopr': out += [dict(r, cut=18), dict(r, cut=35)]            # cut off while it plays, half way and near the end of the loop
        if r['kind'] == 'catr': out.append(dict(r, cut=int(r['dry_sec'] - 1)))       # cut off one second before the dry sound ends
    json.dump(out, open('/tmp/claude-0/v2/roomsure/long_rows.json', 'w')); open('/tmp/claude-0/v2/roomsure/long_list.txt', 'w').write(''.join(r['path'] + ('\t%d' % r['cut'] if r['cut'] else '') + '\n' for r in out))
    import collections; print(len(jobs), 'jobs,', len(rows), 'sounds,', len(out), 'looks', dict(collections.Counter(r['kind'] for r in rows)), 'seconds to look at: %.0f' % sum(r['cut'] or r['sec'] for r in out))
