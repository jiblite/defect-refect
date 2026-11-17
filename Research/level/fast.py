# several builds of drlevel on the same cases of long.py's sets, whole-sound scores only (the sounds are made once per case):
#   python3 fast.py out.jsonl tag=path/to/drlevel [tag=...] -- mode:ids|tog|gat|trm|all ...      (SEED as long.py)
import sys, os, json, subprocess, tempfile, shutil, multiprocessing as mp
import numpy as np, soundfile as sf
import long as L
import sub as S
sys.path.insert(0, '/tmp/claude-0/v2/bench'); import score
def case(a):
    exes, (exe0, mode, i, name, seconds, same) = a
    rng = np.random.default_rng(L.SEED + dict(wet=9000, part=11000, drift=13000, slow=15000)[mode] + i); n = int(seconds * L.SR)
    fn, fam, par, live = L.effect_of(mode, name, rng, n); dry, kind, used = L.long_sound(rng, seconds, same); dryb, kindb, usedb = L.long_sound(rng, seconds, same); wet = fn(dry); wetb = fn(dryb)
    d = tempfile.mkdtemp(dir='/dev/shm'); pa, pb, po, pq = (os.path.join(d, f) for f in ('a.wav', 'b.wav', 'o.wav', 'q.wav')); sf.write(pa, wet, L.SR, subtype='FLOAT'); sf.write(pb, dryb, L.SR, subtype='FLOAT')
    sa = score.Scorer(dry); dw = sa(wet); sb = score.Scorer(wetb); db = sb(dryb); rd = lambda p: sf.read(p, dtype='float64', always_2d=True)[0][:n]
    row = dict(i=i, mode=mode, effect=name, fam=fam, seconds=seconds, sound=kind, v={})
    for tag, exe in exes:
        r = subprocess.run([exe, 'both', pa, pb, po, pq], capture_output=True, text=True)
        if r.returncode: row['v'][tag] = dict(error=r.stderr[-200:]); continue
        M = json.loads(r.stdout); v = dict(found=M['found'], kind=M['kind'] if M['found'] else '', steps=M['steps'], score=M['score'] if M['found'] else None, rate=M['rate'], depth=M['depth'], carrying=M['carrying'], sounding=M['sounding'], steady=M.get('steady'), spans=M.get('spans'), took=M['took'],
                                           pieces=[(p['use'], p.get('guided'), p.get('clocked'), round(p['rate'], 4) if p.get('rate') else None, p.get('edges')) for p in M['pieces']])
        if M['found']:
            out = rd(po); v['out'] = 100.0 * (1.0 - sa(out) / max(dw, 1e-9))
            if live is not None: v['out_off'] = float(10 * np.log10((((out - wet) * (1 - live[:, None])) ** 2).sum() / max(((dry * (1 - live[:, None])) ** 2).sum(), 1e-20) + 1e-20))
            v['onto'] = 100.0 * (1.0 - sb(rd(pq)) / max(db, 1e-9))
        else: v['out'] = 0.0; v['onto'] = 0.0
        row['v'][tag] = v
    shutil.rmtree(d); return row
if __name__ == '__main__':
    out = sys.argv[1]; k = sys.argv.index('--'); exes = [tuple(a.split('=', 1)) for a in sys.argv[2:k]]; jobs = []
    for a in sys.argv[k + 1:]:
        mode, ids = a.split(':'); fams = {'tog': ('trm', 'gat'), 'gat': ('gat',), 'trm': ('trm',)}.get(ids); want = None if ids == 'all' or fams is not None else {int(v) for v in ids.split(',')}
        for j in S.jobs_of(exes[0][1], mode):
            if fams is not None:
                if j[3].startswith(fams) or (mode == 'drift' and j[3] == 'trem' and 'trm' in fams): jobs.append(j)
            elif want is None or j[2] in want: jobs.append(j)
    with mp.Pool(int(os.environ.get('JOBS', '4'))) as pool, open(out, 'w') as f:
        for r in pool.imap_unordered(case, [(exes, j) for j in jobs]): f.write(json.dumps(r) + '\n'); f.flush()
