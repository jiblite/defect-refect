# The long reverbs of long_score.py in absolute terms: how far the sound is from its true dry part untouched (dB per cell
# of the scoreboard's pictures), and how much closer or further the REVERB Amount fader brings it.
#   python3 long_abs.py tag      (reads long_score_<tag>_*.jsonl; writes long_base.json once)
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, soundfile as sf
D = '/tmp/claude-0/v2/roomsure/'; B = D + 'long_base.json'
if not os.path.exists(B):
    import score
    T = json.load(open(D + 'long_truth.json')); base = {}
    for r in T:
        wet = sf.read(r['path'], dtype='float64', always_2d=True)[0]; direct = sf.read(r['truth'], dtype='float64', always_2d=True)[0]; base[r['name']] = float(score.Scorer(direct)(wet))
    json.dump(base, open(B, 'w'))
base = json.load(open(B))
for tag in sys.argv[1:]:
    R = [json.loads(l) for f in sorted(glob.glob(D + 'long_score_%s_*.jsonl' % tag)) for l in open(f)]; R = [r for r in R if r['form'] == 'rings out']
    k2 = 'half' if 'half' in R[0] else 'bold'        # a run with FADER=046: far left is the answer ('out') and half way left half of it ('half')
    b = np.array([base[r['name']] for r in R]); o = np.array([-base[r['name']] * r['out'] / 100.0 for r in R]); f = np.array([-base[r['name']] * r[k2] / 100.0 for r in R]); light = np.array([r['drr'] >= 5 for r in R])
    print('[%s] %d long reverbs; untouched distance: median %.2f dB (light %.2f, heavy %.2f)' % (tag, len(R), np.median(b), np.median(b[light]), np.median(b[~light])))
    for lab, d in ((('the answer in full (far left since 0.4.6)', o), ('half of the answer (half way left since 0.4.6)', f)) if k2 == 'half' else (('the answer in full (half way left up to 0.4.5)', o), ('the bolder far left of 0.4.1 to 0.4.5', f))):
        for l2, s in (('all', np.ones(len(R), bool)), ('light', light), ('heavy', ~light)):
            print('   %-48s %-5s n=%2d | change: median %+.2f dB | closer by 0.5 dB+: %2d | further: %2d, by 0.25 dB+: %2d, by 0.5 dB+: %2d, by 1 dB+: %2d (worst %+.2f)' % (lab, l2, s.sum(), np.median(d[s]), (d[s] <= -0.5).sum(), (d[s] > 0).sum(), (d[s] >= 0.25).sum(), (d[s] >= 0.5).sum(), (d[s] >= 1.0).sum(), d[s].max()))
