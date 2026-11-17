# The final engine once more on every clean corpus sound (the statistics in stats_clean.jsonl were gathered before the last
# level test existed): is anything reported? Writes recheck_clean.json (id, kind, split, found, lines that pass, lines followed up).
import sys, os, json, glob, numpy as np, soundfile as sf
os.chdir('/tmp/claude-0/v2/sweep'); sys.path.insert(0, '/tmp/claude-0/v2/sweep')
import sweep
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); rows = []
for fp in sorted(glob.glob('/tmp/claude-0/v2/corpus/dry/*.wav')):
    name = os.path.splitext(os.path.basename(fp))[0]; x, sr = sf.read(fp, dtype='float64', always_2d=True); x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
    if not np.abs(x).max() > 1e-4: continue
    info = []; F = sweep.find(x, info=info)
    rows.append(dict(id=name, kind=meta.get(name, {}).get('kind', '?'), split=meta.get(name, {}).get('split', '?'), found=F is not None, passing=sum(1 for cl in info for f in cl if sweep.accept(f)), followed=sum(1 for cl in info for f in cl if f.get('g', 0) != 0)))
    json.dump(rows, open('recheck_clean.json', 'w'))
for sp in ('dev', 'test'):
    rr = [r for r in rows if r['split'] == sp]; print('clean corpus %-4s: reported on %d of %d sounds; single lines that pass: %d of %d followed up  %s' % (sp, sum(r['found'] for r in rr), len(rr), sum(r['passing'] for r in rr), sum(r['followed'] for r in rr), ' '.join(r['id'] for r in rr if r['found'] or r['passing'])))
