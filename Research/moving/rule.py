# What the moving-tone engine (moving.find) says on every sound of the scoreboard (19 families of outside effects, the wet
# sounds) and on the clean corpus: found or not, which kind, how sure, how much of the sound. For the sureness rule.
#   python3 rule.py model out.json [family ...]         (no family: all 19 and 'clean')
#   python3 rule.py --show out.json [cut]                the table: per family how many sounds show PHASER / FILTER SWEEP at the cut (0.65)
import sys, os, json, glob, numpy as np
sys.path[:0] = ['/tmp/claude-0/v2/moving', '/tmp/claude-0/v2/sweep']
SETS = '/tmp/claude-0/v2/bench/sets/'

def show(path, cut=0.65):
    R = json.load(open(path)); fams = sorted({r['fam'] for r in R}, key=lambda f: (f not in ('phaser', 'filtersweep'), f))
    for f in fams:
        rows = [r for r in R if r['fam'] == f]; on = [r for r in rows if r['found'] and r['sure'] > cut]
        print('%-12s %3d sounds | found %3d | over %.2f: %3d (sweep %d, phaser %d) | sureness of the found: %s' % (f, len(rows), sum(r['found'] for r in rows), cut, len(on), sum(r['kind'] == 'filtersweep' for r in on), sum(r['kind'] == 'phaser' for r in on),
              ' '.join('%.2f' % r['sure'] for r in sorted(rows, key=lambda r: -r['sure']) if r['found'])[:150]))

if __name__ == '__main__':
    if sys.argv[1] == '--show': show(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.65); sys.exit()
    import torch, soundfile as sf, moving; torch.set_num_threads(int(os.environ.get('THREADS', '4')))
    from lab import cases, load
    model = moving.model_of('/tmp/claude-0/v2/moving/%s.pt' % sys.argv[1]); out = sys.argv[2]; fams = sys.argv[3:] or sorted(os.listdir(SETS)) + ['clean']; R = []
    def one(fam, name, x):
        F = moving.find(model, x); T = F['T']; inside = float(F['gate'].sum()) / T
        R.append(dict(fam=fam, name=name, found=F['found'], kind=F['kind'], sure=F['sure'], share=inside, period=F['period'], on=float(F['on'].mean()), top=float(np.sort(F['on'])[-max(1, T // 10):].mean()), rms=float(np.sqrt((F['G'] ** 2).mean()))))
    for fam in fams:
        if fam == 'clean':
            meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
            for cid in sorted(meta):
                x = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % cid, dtype='float64', always_2d=True)[0]; x = x if x.shape[1] > 1 else np.repeat(x, 2, 1); one('clean', cid, x[:, :2])
        else:
            for d in cases(fam):
                for w in 'AB': c = load(d, w); one(fam, c['name'] + '_' + w, c['wet'])
        json.dump(R, open(out, 'w')); print(fam, len(R), flush=True)
    show(out)
