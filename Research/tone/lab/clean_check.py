# The default engine on every clean training clip: how often is a reflection reported where there is none, and what does flattening do then?
import sys, os, glob, json, time
os.environ['OMP_NUM_THREADS'] = '1'
sys.path[:0] = ['/tmp/claude-0/v2/tone', '/tmp/claude-0/v2/bench']
import numpy as np, soundfile as sf
from multiprocessing import Pool
def one(p):
    import tone, score
    x = sf.read(p, always_2d=True)[0].astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    info = []; g, c = tone.contour(x[:, :2], info=info)
    if not info or not info[0]['echoes']: return None
    return dict(echoes=[(float(d), float(gn)) for d, gn in info[0]['echoes']], harm=float(score.Scorer(x[:, :2])(tone.apply_curve(x[:, :2], g, c, 0.0))), clip=os.path.basename(p)[:-5])
if __name__ == '__main__':
    files = sorted(glob.glob('/tmp/claude-0/v2/train/dry/tr_*.flac')); kinds = json.load(open('/tmp/claude-0/v2/train/dry_meta.json')); t0 = time.time()
    with Pool(2) as pool: R = pool.map(one, files, chunksize=8)
    hits = [r for r in R if r]; print('%d clean clips, a reflection reported on %d (%.1f%%), %.0f s' % (len(files), len(hits), 100 * len(hits) / len(files), time.time() - t0))
    for r in hits: print('   kind %-16s %s  harm %.2f' % (kinds[r['clip']], ' '.join('%.1f ms gain %+.2f' % (d / 48, g) for d, g in r['echoes']), r['harm']))
    json.dump([dict(kind=kinds[r['clip']], echoes=r['echoes'], harm=r['harm']) for r in hits], open('/tmp/claude-0/v2/tone/lab/work/clean_check.json', 'w'))
