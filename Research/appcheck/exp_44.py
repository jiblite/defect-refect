# As exp_all.py, for files that are not at 48 kHz (the app's loader and the test renderer now share one conversion).
# y10 .. y12 move MOVEMENT's faders on such files.
#   python3 exp_44.py run A|B | compare
import sys, exp_all as E
MC = E.MC
E.CASES = [
    ('y1', MC + 'both44.wav', None, [], 1, []),
    ('y2', MC + 'both44.wav', None, [534], 0, ['stutter.amount']),
    ('y3', MC + 'both44.wav', None, [534], 2, ['stutter.amount']),
    ('y4', MC + 'both44.wav', MC + 'sec44.wav', [], 1, []),
    ('y5', MC + 'stut96.wav', None, [422], 2, ['stutter.amount']),
    ('y6', MC + 'stut96.wav', None, [446], 0, ['stutter.slice']),
    ('y7', MC + 'crush96.wav', MC + 'sec44.wav', [422, 446, 470], 2, ['bitcrush.amount', 'bitcrush.bits', 'bitcrush.down']),
    ('y8', E.X + 'tone96.wav', None, [], 1, []),
    ('y9', MC + 'soft44.wav', None, [], 1, []),
    ('y10', MC + 'both44.wav', None, [], 1, [], [E.LOUD, E.TONE, E.SPEC], 0),
    ('y11', MC + 'stut96.wav', MC + 'sec44.wav', [], 1, [], [E.PITCH, E.LOUD, E.TONE, E.STEREO, E.SPEC], 2),
    ('y12', MC + 'both44.wav', None, [], 1, [], [E.PITCH], 2),
]
E.CASES = E.padded(E.CASES)
if __name__ == '__main__': {'run': lambda: E.run(sys.argv[2]), 'compare': E.compare}[sys.argv[1]]()
