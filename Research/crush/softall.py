# The crush prototype on its own model of a crusher that mixes plain sound in, on every clip of the corpus (one setting
# that softsweep.py finds on all of its three clips: 7-bit steps, nine tenths crushed, a hold of three samples, blend 0.3).
# Per clip: found or not, what was read, and what the clip is like (its peak in steps of the grid, the share of its
# samples that sit at one of its two extreme values: a clipped sound).
#   python3 softall.py [jobs]      writes /tmp/claude-0/v2/newfx/softall.jsonl (ids of the corpus: stays out of the repo)
import sys, os, json, glob, warnings, numpy as np, soundfile as sf
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import crush
DRY = '/tmp/claude-0/v2/corpus/dry/'; OUT = '/tmp/claude-0/v2/newfx/softall.jsonl'
F = dict(kind='soft', found=True, mix=0.9, soft=0.3, step=[2.0 / 2 ** 7] * 2, offset=[0.0, 0.0], P=3.0, c=0.0)
def one(p):
    x = sf.read(p, dtype='float64', always_2d=True)[0]
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    pk = float(np.abs(x).max()); top = float(max((x[:, ch] == x[:, ch].max()).mean() + (x[:, ch] == x[:, ch].min()).mean() for ch in range(2)))
    u, cnt = np.unique(x[:, 0][x[:, 0] != 0.0], return_counts=True); most = float(cnt.max() / max(len(x), 1)) if len(cnt) else 0.0
    f = crush.find(crush.soft_forward(x, F)); t = f.get('soft_try')
    ok = bool(f.get('found')) and f.get('kind') == 'soft'
    return dict(id=os.path.basename(p)[:-4], sec=len(x) / 48000.0, peak=pk, steps=pk / F['step'][0], top=top, most=most, found=bool(f.get('found')), kind=f.get('kind'), soft=ok,
                P=f.get('P') if ok else None, mix=f.get('mix') if ok else None, blend=f.get('soft') if ok else None, tried=t is not None,
                dense=None if t is None else t['dense'], inside=None if t is None else t['inside'], half=None if t is None else t['half'])
if __name__ == '__main__':
    ps = sorted(glob.glob(DRY + '*.wav')); rows = []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as ex, open(OUT, 'w') as o:
        for r in ex.map(one, ps): rows.append(r); o.write(json.dumps(r) + '\n'); o.flush()
    n = len(rows); s = [r for r in rows if r['soft']]; h = [r for r in s if abs(r['P'] - 3.0) < 1e-6]
    print('%d clips: found as a mixing crusher on %d, with the hold of three samples on %d; share within 0.03 on %d, blend within 0.05 on %d' % (n, len(s), len(h), sum(abs(r['mix'] - 0.9) <= 0.03 for r in s), sum(abs(r['blend'] - 0.3) <= 0.05 for r in s)))
    for name, sel in (('more than 1 %% of the samples at an extreme value', lambda r: r['top'] > 0.01), ('one value other than zero on more than 1 %% of the samples', lambda r: r['most'] > 0.01),
                      ('peak under 20 steps of the grid', lambda r: r['steps'] < 20), ('none of these', lambda r: r['top'] <= 0.01 and r['most'] <= 0.01 and r['steps'] >= 20)):
        g = [r for r in rows if sel(r)]; print('  %-62s %3d clips, found on %3d' % (name % () if '%%' in name else name, len(g), sum(r['soft'] for r in g)))
    print('  found as something else (not the mixing kind):', sum(r['found'] and not r['soft'] for r in rows))
