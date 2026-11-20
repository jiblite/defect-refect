# A found cut on a second sound, and the cut taken out of the first: the app's code (the render test of Tests/render.cpp)
# against build 0.5.1's on the twelve cases through outside hard clippers (mk_clip.py: clip_sox, clip_pedalboard,
# clip_ff_hard), each with a second sound through the same clipper, driven as hard against its own peak.
#   python3 second.py new_drrender old_drrender [out.json]
# Score: 100 = the second sound through the real clipper (the first sound before it), 0 = untouched, under 0 = worse.
# The folders hold audio and stay on the machine (DR_SETS3, DR_DECLIP_TMP); nothing but case names and numbers is printed.
import sys, os, re, json, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, os.environ.get('DR_BENCH', '/tmp/claude-0/v2/bench')); sys.path.insert(0, os.environ.get('DR_PROTO', '/tmp/claude-0/v2/proto'))
import truth, score
NEW, OLD = sys.argv[1], sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else None; RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Resources')
SETS3 = os.environ.get('DR_SETS3', '/tmp/claude-0/v2/declip/sets3/clip/'); T = os.environ.get('DR_DECLIP_TMP', '/tmp/claude-0/v2/declip/second/'); os.makedirs(T, exist_ok=True)
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split(); BEND = ('distortion', 'saturation')
others = [k + '.amount=0' for k in ALL if k not in BEND]; tot = {}
for n in sorted(d for d in os.listdir(SETS3) if d.startswith(('clip_sox_', 'clip_pedalboard_', 'clip_ff_hard_'))):
    cA, cB = truth.load(SETS3 + n + '/', 'A'), truth.load(SETS3 + n + '/', 'B'); sf.write(T + 'a.wav', cA['wet'], 48000, subtype='FLOAT'); sf.write(T + 'b.wav', cB['dry'], 48000, subtype='FLOAT'); res = {}
    for name, exe in (('old', OLD), ('new', NEW)):
        r = subprocess.run([exe, RES, T + 'a.wav', T + 'b.wav', T + 'o.wav'] + others + ['--', '-', T + 'y.wav', 'distortion.amount=0', 'saturation.amount=0'], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
        shown = [s for s, _ in re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', r.stdout, re.M) if s in BEND]; bend = (re.search(r'^bend: (.*)$', r.stdout, re.M) or [None, ''])[1]; top = re.search(r' top_db=([0-9.eE+-]+)', bend)
        o = sf.read(T + 'o.wav', dtype='float64', always_2d=True)[0][:len(cB['dry'])]; y = sf.read(T + 'y.wav', dtype='float64', always_2d=True)[0][:len(cA['wet'])]
        res[name] = dict(onto=score.removed(cB['wet'], cB['dry'], o)[0], out=score.removed(cA['dry'], cA['wet'], y)[0], shown=shown, rebuilt='rebuilt=1' in bend, top=float(top.group(1)) if top else None, over=float(np.abs(y).max() / np.abs(cA['wet']).max()))
    res['drive'] = cA['meta']['params']['drive_db']; tot[n] = res; a, b = res['old'], res['new']
    print('%-20s driven %5.2f dB, read %5.2f | second sound: 0.5.1 %7.1f (%s) | new %7.1f (%s%s) || taken out: 0.5.1 %7.1f | new %7.1f | highest value against the file\'s %.2f' % (n, res['drive'], b['top'] or 0, a['onto'], ','.join(a['shown']) or 'not shown',
          b['onto'], ','.join(b['shown']) or 'not shown', ', rebuilt' if b['rebuilt'] else '', a['out'], b['out'], b['over']), flush=True)
for name, S in (('rebuilt', [n for n in tot if tot[n]['new']['rebuilt']]), ('not rebuilt', [n for n in tot if not tot[n]['new']['rebuilt']]), ('all', list(tot))):
    m = lambda w, k: np.mean([tot[n][w][k] for n in S]); d = [tot[n]['new']['onto'] - tot[n]['old']['onto'] for n in S]
    print('%-12s (%d): second sound 0.5.1 mean %.1f, new %.1f | new worse than 0.5.1 by 5 or more: %d, better by 5 or more: %d | taken out 0.5.1 %.1f, new %.1f' % (name, len(S), m('old', 'onto'), m('new', 'onto'), sum(v <= -5 for v in d), sum(v >= 5 for v in d), m('old', 'out'), m('new', 'out')))
if OUT: json.dump(tot, open(OUT, 'w'), indent=1)
