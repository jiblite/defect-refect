# A clipped file whose cut sits at full scale, as most real ones do: does the app's limiter work on the rebuilt tops?
# Each case goes through the render test (Tests/render.cpp) as it is, brought to full scale (float), and brought to
# full scale as a 16 bit file; DISTORTION (or SATURATION) at far left, and once with all faders in the middle.
#   python3 fullscale.py drrender
# Score: 100 = the sound before the effect, 0 = untouched. Prints case names and numbers only.
import sys, os, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, os.environ.get('DR_BENCH', '/tmp/claude-0/v2/bench')); sys.path.insert(0, os.environ.get('DR_PROTO', '/tmp/claude-0/v2/proto'))
import score, truth
EXE = sys.argv[1]; RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Resources'); T = os.environ.get('DR_DECLIP_TMP', '/tmp/claude-0/v2/declip/second/'); os.makedirs(T, exist_ok=True)
SETS = os.environ.get('DR_SETS', '/tmp/claude-0/v2/bench/sets/'); SETS3 = os.environ.get('DR_SETS3', '/tmp/claude-0/v2/declip/sets3/clip/')
for case, d in (('clip_sox_01', SETS3 + 'clip_sox_01/'), ('clip_sox_02', SETS3 + 'clip_sox_02/'), ('dist_pedalboard_00', SETS + 'distortion/dist_pedalboard_00/'), ('sat_tap_tube_01', SETS + 'saturation/sat_tap_tube_01/'), ('dist_sox_01', SETS + 'distortion/dist_sox_01/')):
    c = truth.load(d, 'A'); wet, dry = c['wet'], c['dry']; pk = np.abs(wet).max(); line = []
    for name, g, sub in (('as it is', 1.0, 'FLOAT'), ('at full scale', 1.0 / pk, 'FLOAT'), ('at full scale, 16 bit', 1.0 / pk, 'PCM_16')):
        sf.write(T + 'in.wav', np.clip(wet * g, -1.0, 1.0), 48000, subtype=sub)
        r = subprocess.run([EXE, RES, T + 'in.wav', '-', T + 'out.wav', 'distortion.amount=0', 'saturation.amount=0', '--', '-', T + 'mid.wav'], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
        o = sf.read(T + 'out.wav', dtype='float64', always_2d=True)[0][:len(wet)]; mid = sf.read(T + 'mid.wav', dtype='float64', always_2d=True)[0][:len(wet)]
        line.append('%s: highest value in %.3f, out %.3f, score %.1f (faders in the middle: out %.3f)' % (name, np.abs(wet * g).max(), np.abs(o).max(), score.removed(dry, wet, o)[0], np.abs(mid).max()))
    print(case, '|', ' | '.join(line), flush=True)
