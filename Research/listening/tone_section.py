# The page's Tone section (build 0.4.1): three examples rendered by the app's own engine (the command-line tool drrender), as MP3s
# in audio/tone_*.mp3, and the section's HTML in tone.html (build.py puts it in front of the outro).
#   python3 tone_section.py <folder with the test WAVs and out/ renders>
import os, sys, json, html, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); import score
HERE = os.path.dirname(os.path.abspath(__file__)); AUD = os.path.join(HERE, 'audio'); T = sys.argv[1].rstrip('/') + '/'; SR = 48000; A0 = int(1.35 * SR)
def rd(n): x, sr = sf.read(T + n, dtype='float64', always_2d=True); assert sr == SR, n; return x
def mp3(name, x, gain):
    wav = os.path.join(AUD, name + '.wav'); y = np.clip(x[A0:] * gain, -1.0, 1.0); n = len(y); f = min(480, n // 4); y[:f] *= np.linspace(0, 1, f)[:, None]; y[-f:] *= np.linspace(1, 0, f)[:, None]
    sf.write(wav, y, SR, subtype='PCM_24'); subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame', '-b:a', '256k', os.path.join(AUD, name + '.mp3')], check=True); os.remove(wav)
    return 'audio/%s.mp3' % name
def row(cls, label, small, src): return '<div class="row %s"><div class="lab">%s<small>%s</small></div><audio controls preload="none" src="%s"></audio></div>' % (cls, html.escape(label), html.escape(small), src)
def gain(ref, base, out): return 100 * (1 - score.distance(ref, out) / score.distance(ref, base))
def files(tag, d):
    x = {k: rd(v) for k, v in d.items()}; g = 0.89 / max(np.abs(v).max() for v in x.values()); return {k: mp3('tone_%s_%s' % (tag, k), v, g) for k, v in x.items()}, x
E = html.escape; parts = []
# 1. a short reflection, one sound and onto a second
f, x = files('c', dict(wet='tone_comb_03_Awet.wav', out='out/c03_one_removed.wav', dry='tone_comb_03_Adry.wav', b_dry='tone_comb_03_Bdry.wav', b_onto='out/c03_onto_only.wav', b_real='tone_comb_03_Bwet.wav'))
s1 = gain(x['dry'], x['wet'], x['out']); s2 = gain(x['b_real'], x['b_dry'], x['b_onto']); m = json.load(open('/tmp/claude-0/v2/bench/sets/tone/tone_comb_03/meta.json')); pr = eval(m['params']) if isinstance(m['params'], str) else m['params']
parts.append('<section class="panel"><h3>7. A short reflection on a piano melody</h3><div class="meta">A copy of the sound %.1f ms later at %d %% of its level, which gives the hollow colour of a comb filter. The app found it at 1.5 ms and 57 %%. Scores: taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.</div>'
             '<div class="sub">One sound: the COMB FILTER group</div><div class="rows">%s</div><div class="sub">Two sounds: the found reflection on a second sound</div><div class="rows">%s</div></section>' % (
    pr['ms'], round(100 * abs(pr['g'])), s1, s2,
    ''.join([row('a', 'As it came in', 'with the reflection on it', f['wet']), row('a', 'Amount at far left', 'the reflection taken out', f['out']), row('ref', 'For comparison', 'the sound before the reflection was added', f['dry'])]),
    ''.join([row('b', 'Second sound, dry', 'a drum machine pattern', f['b_dry']), row('b', 'With the reflection found above', 'fader in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])])))
# 2. Tone Match, two versions of one recording
f, x = files('p', dict(first='tone_pb_eq_02_Awet.wav', second='tone_pb_eq_02_Adry.wav', match='out/eq02_match_only.wav'))
s3 = gain(x['first'], x['second'], x['match']); m = json.load(open('/tmp/claude-0/v2/bench/sets/tone/tone_pb_eq_02/meta.json')); pr = eval(m['params']) if isinstance(m['params'], str) else m['params']; nb = len(pr['bands'])
parts.append('<section class="panel"><h3>8. Tone Match with two versions of one recording</h3><div class="meta">First sound: one of your recordings with a %d-band EQ on it. Second sound: the same recording without the EQ. The app notices that the two are one recording and works out the curve between them, for each side. Score: <b>%.0f</b> (100 = identical to the first sound).</div>'
             '<div class="rows">%s</div></section>' % (nb, s3, ''.join([row('a', 'First sound', 'with the EQ on it', f['first']), row('b', 'Second sound', 'the same recording without the EQ', f['second']), row('b', 'Second sound with Tone Match', 'Amount in the middle', f['match'])])))
first_eq = f['first']
# 3. Tone Match, two different sounds
f, x = files('q', dict(second='tone_pb_eq_02_Bdry.wav', match='out/eq02_matchB.wav'))
parts.append('<section class="panel"><h3>9. Tone Match with two different sounds</h3><div class="meta">First sound as in example 8. Second sound: another of your recordings. There is no true answer to score against here: the second sound takes the long-term tone of the first in broad strokes (Detail in the middle).</div>'
             '<div class="rows">%s</div></section>' % ''.join([row('a', 'First sound', 'the tone to take', first_eq), row('b', 'Second sound', 'as it is', f['second']), row('b', 'Second sound with Tone Match', 'Amount in the middle', f['match'])]))
head = ('<section><h2>Tone: new in build 0.4.1</h2><p>The tone engine does two things. On one sound it finds short reflections and takes them out with the exact opposite filter. With two sounds, the Tone Match button gives the second sound the tone of the first, as a curve you can see, shape with Amount and Detail, and save with Export IR.</p>'
        '<p>What it cannot do is find a plain EQ on one sound alone. I measured this: from the outside an EQ and the sound\'s own colour look the same, and every rule I tried did more harm than good.</p>'
        '<p>In examples 8 and 9 the other effect groups were switched off (each group now has an On/Off switch). On these sounds the app also showed a filter and a filter sweep that are not there, and with those left on the result is worse than doing nothing. Fewer such false finds is the next piece of work.</p></section>')
open(os.path.join(HERE, 'tone.html'), 'w').write(head + '\n' + '\n'.join(parts))
print('tone section written: scores %.1f %.1f %.1f' % (s1, s2, s3), '|', sum(os.path.getsize(os.path.join(AUD, n)) for n in os.listdir(AUD) if n.startswith('tone_')) // 1024, 'KB of tone audio')
