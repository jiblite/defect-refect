# As exp_all.py, for the groups of build 0.5.1: the fixed filter (LOW-PASS, HIGH-PASS, BAND-PASS FILTER) and the bend
# (DISTORTION, SATURATION), each alone and onto a second sound, both on one sound, next to MOVEMENT's faders, and on
# files at 44.1 kHz (a bend is read off such a file's own samples, so the app and the test renderer must do that alike).
#   python3 exp_051.py prepare | run A|B [tags] | compare [tags]
import sys, exp_all as E
MC = E.MC; Q = '/tmp/claude-0/v2/four/mc/'; E.EXE = '/tmp/claude-0/v2/four/build/drrender_new'
FA, FC, FR = 'filter.amount', 'filter.cutoff', 'filter.resonance'; DA, SA = 'distortion.amount', 'saturation.amount'
E.CASES = [   # tag, first sound, second sound, fader rows pulled, to which end, the same faders for the test renderer[, MOVEMENT's rows pulled, to which end]
    ('f1', 'A:filter/fil_calf_00', None, [422], 0, [FA]),                                   # a low-pass alone: the filter taken out
    ('b1', 'A:distortion/dist_mda_01', None, [422], 0, [DA]),                               # a distortion alone: the bend walked back
    ('f2', 'A:filter/fil_calf_00', None, [446, 470], 2, [FC, FR]),                          # Cutoff and Resonance far right
    ('b2', 'A:distortion/dist_pedalboard_00', None, [422], 2, [DA]),                        # pushed as far again
    ('f3', 'A:filter/fil_sox_00', 'B:filter/fil_sox_00', [422, 446], 2, [FA, FC]),          # a high-pass onto a second sound: Amount and Cutoff far right
    ('b3', 'A:distortion/dist_mda_01', 'B:distortion/dist_mda_01', [], 1, []),              # a distortion onto a second sound
    ('f4', 'A:filter/fil_calf_00', 'B:filter/fil_calf_00', [], 1, []),                      # a low-pass onto a second sound
    ('b4', 'A:saturation/sat_tap_tube_01', 'B:saturation/sat_tap_tube_01', [422], 2, [SA]), # a saturation onto a second sound, pushed as far again
    ('f5', 'A:filter/fil_sox_01', None, [422, 446], 0, [FA, FC]),                           # read as a band-pass: Amount and Cutoff far left
    ('b5', 'A:saturation/sat_tap_tube_01', None, [422], 0, [SA]),                           # a saturation alone: walked back
    ('c1', Q + 'both.wav', None, [422, 486], 0, [SA, FA]),                                  # a filter and a saturation on one sound: both taken out
    ('c2', Q + 'both.wav', MC + 'secMono.wav', [510], 2, [FC], [E.LOUD, E.TONE], 2),        # both onto a mono second sound, Cutoff far right, Loudness and Tone far right
    ('c3', Q + 'both.wav', None, [486, 510], 2, [FA, FC], [E.PITCH], 0),                    # the filter's faders far right, Pitch far left
    ('y21', Q + 'filt44.wav', None, [422], 0, [FA]),                                        # a file at 44.1 kHz: the filter out
    ('y22', Q + 'bend44pcm.wav', None, [422], 0, [DA]),                                     # a file at 44.1 kHz: the bend walked back
    ('y23', Q + 'bend44pcm.wav', MC + 'sec44.wav', [422], 2, [DA]),                         # onto a second sound at 44.1 kHz, pushed as far again
]
E.CASES = E.padded(E.CASES)
if len(sys.argv) > 2 and sys.argv[1] in ('run', 'compare'):
    want = sys.argv[3:] if sys.argv[1] == 'run' else sys.argv[2:]
    if want: E.CASES = [c for c in E.CASES if c[0] in want]
if __name__ == '__main__': {'prepare': E.prepare, 'run': lambda: E.run(sys.argv[2]), 'compare': E.compare}[sys.argv[1]]()
