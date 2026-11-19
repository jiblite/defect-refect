# As exp_all.py, for files that are not at 48 kHz (the app's loader and the test renderer now share one conversion).
#   python3 exp_44.py run A|B | compare
import sys, exp_all as E
MC = E.MC
E.EXE = '/tmp/claude-0/drrender047h'
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
]
if __name__ == '__main__': {'run': lambda: E.run(sys.argv[2]), 'compare': E.compare}[sys.argv[1]]()
