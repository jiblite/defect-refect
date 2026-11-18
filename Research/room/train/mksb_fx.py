# Sounds WITHOUT a reverb as the model sees them, for watching what it takes for room while it trains: the scoreboard's
# cases of the 17 other effect families (wet A and B, 6 s, cut off) -> sb_fx/, and the corpus's clean clips -> sb_cl/.
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, soundfile as sf, common as cm
from multiprocessing import Pool
T = '/tmp/claude-0/v2/train/'; corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
def one(a):
    path, out, info = a
    if os.path.exists(out): return out
    x = sf.read(path, always_2d=True)[0].astype(np.float64); ft = cm.features(x, x); np.savez_compressed(out, info=json.dumps(dict(info, drr=99.0)), **{k: v.astype(np.float16) for k, v in ft.items()}); return out
if __name__ == '__main__':
    jobs = []
    for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/*/*/')):
        fam, case = d.rstrip('/').split('/')[-2:]
        if fam in ('reverb', 'tone'): continue
        m = json.load(open(d + 'meta.json'))
        for w in 'AB': jobs.append((d + w + '_wet.flac', T + 'sb_fx/%s_%s.npz' % (case, w), dict(effect=m['effect'], family=fam, split=m['split'], clip=m[w])))
    for k, v in sorted(corp.items()): jobs.append(('/tmp/claude-0/v2/corpus/dry/%s.wav' % k, T + 'sb_cl/%s.npz' % k, dict(effect='none', family='clean', split=v['split'], clip=k, kind=v['kind'])))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as p: r = p.map(one, jobs, chunksize=8)
    print(len(r))
