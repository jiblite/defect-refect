# The limit of the crush engine's "13 bits or coarser" rule: a quiet recording with 16 bits (every fifth clean clip of
# the corpus at -30 dB, rounded to 16 bits) made louder afterwards and kept as it is. From which gain on is it read as
# a crusher?
#   python3 pcm16b.py
import json, subprocess, numpy as np, soundfile as sf
ids = sorted(json.load(open('/tmp/claude-0/v2/corpus/meta.json'))); T = '/tmp/claude-0/v2/newfx/pcm/'
def find(exe, p): return json.loads(subprocess.run([exe, 'find', p], capture_output=True, text=True).stdout.strip().splitlines()[-1])
tot = {}
for cid in ids[::5]:
    x, sr = sf.read('/tmp/claude-0/v2/corpus/dry/' + cid + '.wav', dtype='float64', always_2d=True)
    q = np.round(x * 10 ** (-30 / 20) * 32768.0) / 32768.0
    for name, g, sub in (('+12 dB float', 4.0, 'FLOAT'), ('+18.1 dB float', 8.0, 'FLOAT'), ('+24 dB float', 16.0, 'FLOAT'), ('+24 dB 24-bit', 16.0, 'PCM_24'), ('+30 dB float', 31.6228, 'FLOAT')):
        sf.write(T + 'y.wav', np.clip(q * g, -1, 1), sr, subtype=sub); c = find('/tmp/claude-0/drcrush_b', T + 'y.wav'); t = tot.setdefault(name, [0, 0, []]); t[0] += 1; t[1] += bool(c['found'])
        if c['found']: t[2].append(round(c.get('bits', 0), 1))
for name, t in tot.items(): print('%-14s %d clips: crush found on %d, bits %s' % (name, t[0], t[1], sorted(set(t[2]))))
