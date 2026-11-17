# a subset of long.py's cases with a given drlevel, in parallel: python3 sub.py path/to/drlevel out.jsonl mode:i,i,... [mode:...]   (SEED as long.py)
import sys, os, json, multiprocessing as mp
import long as L
def jobs_of(exe, mode):
    if mode == 'wet': return [(exe, mode, 8 * e + j, name, L.LENGTHS[j % 4], j < 4) for e, name in enumerate(L.EFFECTS) for j in range(8)]
    if mode == 'part': return [(exe, mode, 4 * e + j, name, L.LENGTHS[1 + j % 3], j < 2) for e, name in enumerate(L.EFFECTS) for j in range(4)]
    if mode == 'drift': return [(exe, mode, 8 * e + j, name, L.LENGTHS[j % 4], j < 4) for e, name in enumerate(('trem', 'pan')) for j in range(8)]
    if mode == 'slow': return [(exe, mode, 8 * e + j, name, L.LENGTHS[j % 4], j < 4) for e, name in enumerate(('pan_sox', 'pan_tap', 'pan_calf')) for j in range(8)]
if __name__ == '__main__':
    exe, out = sys.argv[1], sys.argv[2]; jobs = []
    for a in sys.argv[3:]:
        mode, ids = a.split(':'); fams = {'tog': ('trm', 'gat'), 'gat': ('gat',), 'trm': ('trm',)}.get(ids)
        want = None if ids == 'all' or fams is not None else {int(v) for v in ids.split(',')}
        for j in jobs_of(exe, mode):
            if fams is not None:
                if j[3].startswith(fams) or (mode == 'drift' and j[3] == 'trem' and 'trm' in fams): jobs.append(j)
            elif want is None or j[2] in want: jobs.append(j)
    with mp.Pool(int(os.environ.get('JOBS', '4'))) as pool, open(out, 'w') as f:
        for r in pool.imap_unordered(L.wet_case, jobs): f.write(json.dumps(r) + '\n'); f.flush()
