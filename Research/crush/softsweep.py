# The crush prototype on its own model of a crusher that mixes plain sound in, over holds, shares and blends (three clips).
import sys, json, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path[:0] = ['/home/claude/defect-refect/Research/crush', '/tmp/claude-0/v2/sweep']
import crush
from lab import load
SETS = '/tmp/claude-0/v2/bench/sets/'
clips = [load(SETS + c, 'A')['dry'] for c in ('bitcrush/bit_swh_00/', 'bitcrush/bit_calf_00/', 'bitcrush/bit_pedalboard_01/')]
rows = []
for P in (1.0, 2.0, 3.0, 2.7):
    for m in (0.5, 0.7, 0.9):
        for aa in (0.0, 0.15, 0.3):
            got = []
            for x in clips:
                f = crush.find(crush.soft_forward(x, dict(kind='soft', found=True, mix=m, soft=aa, step=[2.0 / 2 ** 7] * 2, offset=[0.0, 0.0], P=P, c=0.0)))
                ok = bool(f.get('found')) and f.get('kind') == 'soft'
                got.append('no' if not ok else ('hold %g, share %.2f, blend %.2f' % (f['P'], f['mix'], f['soft'])))
            print('hold %.1f share %.1f blend %.2f :' % (P, m, aa), ' | '.join(got), flush=True)
