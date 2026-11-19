# The app's exported files against the test renderer's (Tests/render.cpp, built as drrender) for the same sounds and fader
# positions. The sanitizer build of the app is driven with the mouse (scen.sh exp); every fader used stands at an end of
# its travel or in the middle, where its value is exact. The cases m1 .. m7 move MOVEMENT's faders (alone, next to found
# effects, with two sounds); m8 and m9 pull them while the pitch is still being read (the result has to wait for the
# reading and then be the same); m10 has a second sound of ten samples.
#   python3 exp_all.py prepare | run A|B [tags] | compare [tags]
import sys, os, json, subprocess, numpy as np, soundfile as sf
O = '/tmp/claude-0/drasan/'; X = O + 'x/'; MC = '/tmp/claude-0/v2/newfx/mc/'; SETS = '/tmp/claude-0/v2/bench/sets/'; DRY = '/tmp/claude-0/v2/corpus/dry/'
EXE = '/tmp/claude-0/v2/mv/cc/drrender_v9'; MV = '/tmp/claude-0/v2/mv/cc/'; RES = '/home/claude/defect-refect/Resources'
BC, ST = ['bitcrush.amount', 'bitcrush.bits', 'bitcrush.down'], ['stutter.amount', 'stutter.slice']
SOFT = ['bitcrush.amount', 'bitcrush.bits', 'stutter.slice']      # the mixing crusher has no Sample Rate fader, so STUTTER sits one row higher: its Slice is the row at 534
PITCH, LOUD, TONE, STEREO, SPEC = 422, 446, 470, 494, 518        # MOVEMENT's rows
MID = {PITCH: 'movement.pitch', LOUD: 'movement.loudness', TONE: 'movement.tone', STEREO: 'movement.stereo', SPEC: 'movement.spectral'}
CASES = [   # tag, first sound, second sound, fader rows pulled, to which end, the same faders for the test renderer[, MOVEMENT's rows pulled, to which end]
    ('x1', MC + 'both.wav', None, [422, 534], 0, ['bitcrush.amount', 'stutter.amount']),
    ('x2', MC + 'bothSoft.wav', None, [422, 446, 534], 0, SOFT),
    ('x3', MC + 'both.wav', MC + 'secMono.wav', [], 1, []),
    ('x4', MC + 'bothSoft.wav', 'B:haas/haas_ffmpeg_01', [422, 446, 534], 2, SOFT),
    ('x5', 'A:haas/haas_ffmpeg_01', None, [422], 0, ['haas.amount']),
    ('x6', 'A:ringmod/rng_sox_00', None, [422, 446], 0, ['ringmod.amount', 'ringmod.carrier']),
    ('x7', 'A:width/wid_calf_haasenh_01', None, [422], 0, ['width.amount']),
    ('x8', 'A:haas/haas_ffmpeg_01', 'B:haas/haas_ffmpeg_01', [], 1, []),
    ('x9', 'A:ringmod/rng_sox_00', 'B:ringmod/rng_sox_00', [422, 446], 2, ['ringmod.amount', 'ringmod.carrier']),
    ('x10', 'A:width/wid_calf_haasenh_01', 'B:width/wid_calf_haasenh_01', [], 1, []),
    ('m1', MV + 'vibrato.wav', None, [], 1, [], [PITCH], 0),
    ('m2', MV + 'engine.wav', None, [], 1, [], [PITCH, LOUD, TONE, STEREO, SPEC], 2),
    ('m3', MV + 'vibrato.wav', MV + 'engine.wav', [], 1, [], [PITCH, LOUD, TONE, STEREO, SPEC], 2),
    ('m4', MC + 'both.wav', None, [422, 534], 0, ['bitcrush.amount', 'stutter.amount'], [LOUD, TONE], 0),
    ('m5', MC + 'both.wav', MC + 'secMono.wav', [], 1, [], [LOUD, SPEC], 2),
    ('m6', 'A:haas/haas_ffmpeg_01', None, [], 1, [], [STEREO, SPEC], 0),
    ('m7', MV + 'engine.wav', MV + 'zaps_apart.wav', [], 1, [], [PITCH, LOUD, TONE, STEREO, SPEC], 0),
    ('m8', MV + 'engine.wav', None, [], 1, [], [PITCH], 0, {'EARLY': 'reading'}),
    ('m9', MV + 'vibrato.wav', MV + 'engine.wav', [], 1, [], [PITCH, LOUD, TONE, STEREO, SPEC], 2, {'EARLY': 'reading'}),
    ('m10', MC + 'both.wav', MC + 'secTen.wav', [], 1, [], [PITCH, LOUD, TONE, STEREO, SPEC], 2),
]
def padded(cases): return [(c + ([], 1, {}))[:9] if len(c) < 8 else (c + ({},))[:9] for c in cases]
CASES = padded(CASES)
if len(sys.argv) > 2 and sys.argv[1] in ('run', 'compare'):
    want = sys.argv[3:] if sys.argv[1] == 'run' else sys.argv[2:]
    if want: CASES = [c for c in CASES if c[0] in want]
END = {0: 'left', 1: 'middle', 2: 'right'}
def path(tag, which, spec):
    if spec is None: return None
    if spec[:2] not in ('A:', 'B:'): return spec
    return X + '%s_%s.wav' % (tag, which)
def prepare():
    os.makedirs(X, exist_ok=True)
    for tag, a, b, rows, end, ids, mrows, mend, extra in CASES:
        for which, spec in (('first', a), ('second', b)):
            if spec is None or spec[:2] not in ('A:', 'B:'): continue
            d = SETS + spec[2:] + '/'
            if spec[0] == 'A': y, sr = sf.read(d + 'A_wet.flac', always_2d=True)
            else: y, sr = sf.read(DRY + json.load(open(d + 'meta.json'))['B'] + '.wav', always_2d=True)
            assert sr == 48000; sf.write(path(tag, which, spec), y, sr, subtype='FLOAT')
    print('prepared')
def run(lane):
    mine = CASES[0::2] if lane == 'A' else CASES[1::2]
    for k, (tag, a, b, rows, end, ids, mrows, mend, extra) in enumerate(mine):
        env = dict(os.environ, ROWS=' '.join(map(str, rows)), END=END[end], MROWS=' '.join(map(str, mrows)), MEND=END[mend], **extra)
        cmd = [O + 'scen.sh', str((110 if lane == 'A' else 140) + k), tag, 'exp', path(tag, 'first', a)] + ([path(tag, 'second', b)] if b else [])
        r = subprocess.run(cmd, env=env, capture_output=True, text=True); open(O + 'scen_%s.log' % tag, 'w').write(r.stdout + r.stderr)
        print(tag, r.stdout.strip().split('\n')[-2:], flush=True)
def compare():
    for tag, a, b, rows, end, ids, mrows, mend, extra in CASES:
        out = X + tag + '_ref.wav'; fad = ['%s=%d' % (i, end) for i in ids] + ['%s=%d' % (MID[y], mend) for y in mrows]
        r = subprocess.run([EXE, RES, path(tag, 'first', a), path(tag, 'second', b) or '-', out] + fad, capture_output=True, text=True)
        shown = [l for l in r.stdout.split('\n') if l.startswith('shown') or 'groups' in l][:2]
        e = O + 'export_%s.wav' % tag
        if not os.path.exists(e): print(tag, 'NO EXPORT'); continue
        if not os.path.exists(out): print(tag, 'NO REFERENCE', r.stdout[-300:]); continue
        u, sr = sf.read(e, always_2d=True); v, sr2 = sf.read(out, always_2d=True); info = sf.info(e)
        n = min(len(u), len(v)); g = float(np.sum(u[:n] * v[:n]) / max(np.sum(v[:n] ** 2), 1e-30))
        res = 10 * np.log10(max(np.sum((u[:n] - g * v[:n]) ** 2), 1e-30) / max(np.sum(u[:n] ** 2), 1e-30))
        log = open(O + 'app_%s.log' % tag).read(); san = log.count('ERROR: AddressSanitizer') + log.count('runtime error')
        print('%-4s %s %d Hz  lengths %d %d  gain %.6f  difference %.1f dB  level %.1f dB  sanitizer lines %d' % (tag, info.subtype, sr, len(u), len(v), g, res, 10 * np.log10(np.mean(u ** 2) + 1e-30), san))
if __name__ == '__main__': {'prepare': prepare, 'run': lambda: run(sys.argv[2]), 'compare': compare}[sys.argv[1]]()
