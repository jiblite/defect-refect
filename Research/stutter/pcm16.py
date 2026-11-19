# Wrong finds on ordinary file formats: every clean clip of the corpus as a 16-bit file (plain rounding and with dither),
# as a 16-bit file at -30 dB (a quiet recording), and as a 24-bit file; what the crush and the stutter engine say.
#   python3 pcm16.py
import sys, os, json, subprocess, numpy as np, soundfile as sf
ids = sorted(json.load(open('/tmp/claude-0/v2/corpus/meta.json'))); rng = np.random.default_rng(16); T = '/tmp/claude-0/v2/newfx/pcm/'
def find(exe, p): return json.loads(subprocess.run([exe, 'find', p], capture_output=True, text=True).stdout.strip().splitlines()[-1])
tot = {}
for cid in ids:
    x, sr = sf.read('/tmp/claude-0/v2/corpus/dry/' + cid + '.wav', dtype='float64', always_2d=True)
    forms = {'16': (x, 'PCM_16'), '16 dither': (x + (rng.random(x.shape) - rng.random(x.shape)) / 32768.0, 'PCM_16'), '16 quiet': (x * 10 ** (-30 / 20), 'PCM_16'), '24': (x, 'PCM_24')}
    for name, (y, sub) in forms.items():
        sf.write(T + 'x.wav', np.clip(y, -1, 1), sr, subtype=sub)
        c = find('/tmp/claude-0/drcrush_b', T + 'x.wav'); s = find('/tmp/claude-0/drstutter_b', T + 'x.wav'); t = tot.setdefault(name, [0, 0, 0])
        t[0] += 1; t[1] += bool(c['found']); t[2] += bool(s['found'])
        if c['found'] or s['found']: print(name, cid if not cid.startswith('real_') else 'real_XX', 'crush', c['found'], c.get('kind'), round(c.get('bits', 0), 2), 'stutter', s['found'], flush=True)
for name, t in tot.items(): print('%-10s %d clips: crush found on %d, stutter on %d' % (name, t[0], t[1], t[2]))
