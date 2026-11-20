# As exp_all.py, for the declipper of build 0.5.2: sounds whose tops were cut flat (a plain cut, a soft flat top), alone
# with the fader at far left (the tops rebuilt) and at far right, onto a second sound, a cut the declipper leaves alone,
# a file cut into full scale, a mono file, a long one, files at 44.1 and 96 kHz (the tops are flat only in the file's own
# samples, so the app and the test renderer must both read them there), next to MOVEMENT's faders, and the sound of build
# 0.5.1 that shows a filter and a saturation. The made sounds come from Research/declip/mkcheck.py.
#   python3 exp_052.py prepare | run A|B [tags] | compare [tags]
import sys, os, exp_all as E
MC = E.MC; D = os.environ.get('DR_DECLIP_MC', '/tmp/claude-0/v2/declip/mc/'); Q = '/tmp/claude-0/v2/four/mc/'; E.EXE = '/tmp/claude-0/v2/declip/bin/drrender_052'
FA, FC = 'filter.amount', 'filter.cutoff'; DA, SA = 'distortion.amount', 'saturation.amount'
E.CASES = [   # tag, first sound, second sound, fader rows pulled, to which end, the same faders for the test renderer[, MOVEMENT's rows pulled, to which end]
    ('d1', D + 'cut.wav', None, [422], 0, [DA]),                                    # a plain cut alone: the tops rebuilt
    ('d2', D + 'soft.wav', None, [422], 0, [DA]),                                   # a soft flat top alone: rebuilt in two passes
    ('d3', D + 'cut.wav', MC + 'secMono.wav', [], 1, []),                           # a rebuilt cut onto a mono second sound, the fader in the middle
    ('d4', D + 'soft.wav', MC + 'secMono.wav', [422], 2, [DA]),                     # a soft flat top onto a second sound, as far again
    ('d5', D + 'cut.wav', None, [422], 2, [DA]),                                    # a plain cut alone, cut as deep again
    ('d6', D + 'cutNoise.wav', None, [422], 0, [SA]),                               # a cut the self-test refuses (shown as SATURATION): walked back as build 0.5.1 did
    ('d7', D + 'cutFull.wav', None, [422], 0, [DA]),                                # cut into full scale of a 16 bit file: rebuilt, and no value over the file's highest
    ('d8', D + 'cutMono.wav', MC + 'sec44.wav', [422], 2, [DA]),                    # a mono file with a cut onto a second sound at 44.1 kHz, as far again
    ('d9', D + 'cutTwo.wav', None, [422], 0, [DA]),                                 # another level of cut on each channel
    ('d10', D + 'softLong.wav', None, [422], 0, [DA]),                              # a minute of soft flat tops: the first pass on parts of the sound
    ('d11', D + 'cut.wav', None, [422], 0, [DA], [E.LOUD, E.TONE], 0),              # the tops rebuilt, MOVEMENT's Loudness and Tone far left
    ('d12', D + 'softMono.wav', None, [422], 0, [DA], [E.PITCH], 2),                # a mono soft flat top that is read with 41 %: not shown at the default Sureness, so only Pitch moves the sound
    ('d13', D + 'softMonoP.wav', None, [422], 0, [DA], [E.PITCH], 2),               # a mono soft flat top that is shown: rebuilt, Pitch far right
    ('d14', D + 'cutNoiseUp.wav', None, [422], 0, [DA]),                            # a dense sound cut on one side only, refused: the cut half stays as it is, the other is walked back
    ('c1', Q + 'both.wav', None, [422, 486], 0, [SA, FA]),                          # build 0.5.1's sound with a filter and a saturation: both taken out
    ('c2', Q + 'both.wav', MC + 'secMono.wav', [510], 2, [FC], [E.LOUD, E.TONE], 2),    # both onto a mono second sound, Cutoff far right, Loudness and Tone far right
    ('y31', D + 'cut44.wav', None, [422], 0, [DA]),                                 # a file at 44.1 kHz with a cut: rebuilt in its own samples
    ('y32', D + 'cut44pcm.wav', MC + 'sec44.wav', [422], 2, [DA]),                  # a 16 bit file at 44.1 kHz onto a second sound at 44.1 kHz, as far again
    ('y33', D + 'soft96.wav', None, [422], 0, [DA]),                                # a file at 96 kHz with soft flat tops that are read with 41 %: not shown, the export is the file brought to 48 kHz
    ('y35', D + 'soft96P.wav', None, [422], 0, [DA]),                               # a file at 96 kHz with soft flat tops that are shown: rebuilt in its own samples
    ('y34', D + 'cut22.wav', MC + 'secMono.wav', [], 1, []),                        # a file at 22.05 kHz with a cut onto a mono second sound
]
E.CASES = E.padded(E.CASES)
if len(sys.argv) > 2 and sys.argv[1] in ('run', 'compare'):
    want = sys.argv[3:] if sys.argv[1] == 'run' else sys.argv[2:]
    if want: E.CASES = [c for c in E.CASES if c[0] in want]
if __name__ == '__main__': {'prepare': E.prepare, 'run': lambda: E.run(sys.argv[2]), 'compare': E.compare}[sys.argv[1]]()
