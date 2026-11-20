# Outside clippers for the declipper's own test cases: a hard cut and soft curves, none of them this project's code
# (ffmpeg's asoftclip in its nine kinds, pedalboard's Clipping, sox's gain running into full scale). Registered next to the
# scoreboard's outside effects (Research/detect/zoo_fx.py) as the family 'clip'. Each wrapper only sets the level: the sound
# is brought to full scale by its own peak, driven into the clipper by so many dB, and brought back.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'detect'))
import numpy as np, pedalboard as pb
import zoo_fx as Z
from zoo_fx import reg, U, CH, run_ffmpeg, run_sox, SR
KINDS = ['hard', 'tanh', 'atan', 'cubic', 'exp', 'alg', 'quintic', 'sin', 'erf']
def _peak(x):
    a = np.abs(x).ravel(); a = a[a > 0]; return float(np.quantile(a, 0.9999)) if len(a) else 1.0
def _ff(kind):
    def fn(x, p):
        pk = _peak(x); return run_ffmpeg(x / pk, ['-af', 'volume=%.3fdB,asoftclip=type=%s:oversample=%d' % (p['drive_db'], kind, p['oversample'])]) * pk
    return fn
for k in KINDS:
    reg('clip_ff_' + k, 'clip', 'test', (lambda r: dict(drive_db=U(r, 3, 18), oversample=1)) if k == 'hard' else (lambda r: dict(drive_db=U(r, 6, 28), oversample=int(CH(r, [1, 1, 4])))), _ff(k))
def _pb_clip(x, p):
    pk = _peak(x); return pb.Pedalboard([pb.Gain(gain_db=p['drive_db']), pb.Clipping(threshold_db=0.0)])((x / pk).T.astype(np.float32), SR, reset=True).T.astype(np.float64) * pk
reg('clip_pedalboard', 'clip', 'dev', lambda r: dict(drive_db=U(r, 3, 18)), _pb_clip)
def _sox_clip(x, p):
    pk = _peak(x); return run_sox(x / pk, ['gain', '%.3f' % p['drive_db']]) * pk
reg('clip_sox', 'clip', 'dev', lambda r: dict(drive_db=U(r, 3, 18)), _sox_clip)
FAMILY = [('clip_ff_' + k, 3) for k in KINDS] + [('clip_pedalboard', 5), ('clip_sox', 4)]
