# Long cases for the declipper: a case's own effect and settings on its clean sound followed by seven more clean clips
# of the same split (48 s). Written as WAVs (24 bit) for the app's code and the prototype alike. Container only: the
# folder holds audio; no clip names are printed.
#   DR_SETS=/tmp/claude-0/v2/bench/sets/ DR_CORPUS=/tmp/claude-0/v2/corpus/ python3 mk_long.py out_dir clips case_dir [case_dir ...]
import sys, os, json, hashlib
sys.path.insert(0, '/home/claude/defect-refect/Research/detect'); sys.path.insert(0, '/home/claude/defect-refect/Research/declip')
out, clips, dirs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]; sys.argv = sys.argv[:1]
import numpy as np, soundfile as sf
import mkpos, clip_fx, zoo_fx as Z
os.makedirs(out, exist_ok=True)
for d in dirs:
    d = d.rstrip('/') + '/'; m = json.load(open(d + 'meta.json')); name = d.split('/')[-2]; tag = 'new_' if 'sets2' in d else ''
    rng = np.random.default_rng(int(hashlib.md5((name + tag).encode()).hexdigest(), 16) % 2**32); names = sorted(n for n, mm in mkpos.meta.items() if mm['split'] == m['split'] and n != m['A'])
    pick = [m['A']] + [str(v) for v in rng.choice(names, clips - 1, replace=False)]; x = np.vstack([mkpos.dry(n) for n in pick]); w = Z.apply(m['effect'], x, m['params'], m['tail'])[:len(x)]
    pk = np.abs(w).max(); g = 1.0 if pk < 0.98 else 0.98 / pk; pre = os.path.join(out, tag + name + '_long')
    sf.write(pre + '_wet.wav', w * g, 48000, subtype='PCM_24'); sf.write(pre + '_dry.wav', x / max(1.0, np.abs(x).max()), 48000, subtype='PCM_24')
    print(tag + name, m['effect'], 'clips', len(pick), 'kinds', ' '.join(sorted(set('own' if 'real' in n else n.split('_')[0] for n in pick))), 'samples', len(x), 'gain %.3f' % g, flush=True)
