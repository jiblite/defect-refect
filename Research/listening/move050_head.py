# The head of the page's 0.5.0 section (move050_head.html): the texts, with their numbers read from the result files:
# the pitch reader on its test sounds (Research/movement/test_flow.py --app: the lines of the app's reader), what the
# app's reader follows in the owner's 40 recordings (survey.json, by index only), the places where another sound may
# start in the kept curves (kinds.py), and the check that everything else is as in build 0.4.7 (same.py).
#   python3 move050_head.py
# Typed in from Research/movement/test_move.py (the other four faders): OTHER below.
import os, re, json, html
HERE = os.path.dirname(os.path.abspath(__file__)); MV = '/tmp/claude-0/v2/mv/'; E = html.escape
# ---------------------------------------------------------------- the pitch reader on its test sounds
T = {}; name = None
for l in open(MV + 'rd/parity_v9.log'):
    m = re.match(r'^(\S.*?)\s+held ([0-9.]+) \|', l)
    if m: name = m.group(1).strip(); continue
    if l.startswith('   the app') and name:
        held = float(re.search(r'held ([0-9.]+)', l).group(1)); e = re.search(r'err\s+([0-9.]+|nan) gain\s+([0-9.]+|nan) apart\s+([0-9.]+)', l); ap = re.search(r'apart from the prototype: ([0-9.]+) cents at most, followed or not differs on ([0-9.]+)', l)
        T[name] = dict(held=held, err=float(e.group(1)) if e else None, gain=float(e.group(2)) if e else None, proto=float(ap.group(1)), differs=float(ap.group(2))); name = None
assert len(T) == 19 and all(t['proto'] == 0.0 and t['differs'] == 0.0 for t in T.values()), T          # the text says: the app's reader gives the prototype's curves on all of them
pc = lambda v: '%.0f %%' % (100 * v); one = lambda v: '%.1f' % v
def tr(label, key, note=''):
    t = T[key]; g = t['gain']; read = note or ('all of it' if g is not None and g == g and g >= 0.995 else pc(g) if g is not None and g == g else 'nothing to read')
    return '<tr><td>%s</td><td class="n">%s</td><td class="n">%s</td><td class="n">%s</td></tr>' % (E(label), pc(t['held']), one(t['err']) if t['err'] is not None and t['err'] == t['err'] else '', E(read))
assert T['white noise']['held'] == 0 and T['pink noise']['held'] == 0 and T['zaps overlapping']['held'] == 0 and T['tune']['err'] < 2 and T['two tones beating']['err'] < 0.5 and T['engine steady']['err'] < 2
rows = [tr('A vibrato, 30 cents up and down', 'vibrato'), tr('A glide over 600 cents', 'glide'), tr('A pass-by, 300 cents', 'doppler'), tr('A rev over 1200 cents', 'rev'),
        tr('An engine with partials 22 Hz apart, revving over 1200 cents', 'engine'), tr('The same engine under a lot of noise', 'engine in noise'),
        tr('A glide under clicks', 'glide with bursts'), tr('A rev under bursts of noise', 'rev with bursts'),
        tr('An engine that drops 500 cents at once', 'engine with a jump', 'all of it on both sides; the drop itself is not read'),
        tr('Falling zaps with pauses between them', 'zaps apart'), tr('Falling zaps without pauses', 'zaps back to back', '%s; the rest is left in the sound, which is what the error counts' % pc(T['zaps back to back']['gain']))]
assert T['engine with a jump']['gain'] >= 0.995
none = ('<tr><td>Zaps that overlap</td><td class="n">0 %%</td><td class="n"></td><td class="n">nothing is read</td></tr>'
        '<tr><td>White noise, pink noise</td><td class="n">0 %%</td><td class="n"></td><td class="n">nothing is read, as it should be</td></tr>'
        '<tr><td>A steady engine, two steady tones that beat, a tune of steady notes</td><td class="n">%s to %s</td><td class="n">%s to %s</td><td class="n">no movement read, as it should be</td></tr>'
        % ('%.0f' % (100 * min(T[k]['held'] for k in ('engine steady', 'two tones beating', 'tune'))), pc(max(T[k]['held'] for k in ('engine steady', 'two tones beating', 'tune'))).replace(' ', '&nbsp;'),
           one(min(T[k]['err'] for k in ('engine steady', 'two tones beating', 'tune'))), one(max(T[k]['err'] for k in ('engine steady', 'two tones beating', 'tune')))))
# ---------------------------------------------------------------- the owner's recordings
S = json.load(open(MV + 'page/survey.json')); assert len(S) == 40
n80 = sum(r['held'] >= 0.8 for r in S); n40 = sum(0.4 <= r['held'] < 0.8 for r in S); n10 = sum(0.1 <= r['held'] < 0.4 for r in S); n0 = sum(r['held'] < 0.1 for r in S); assert n80 + n40 + n10 + n0 == 40
kinds = [l for l in open(MV + 'page/kinds_y.txt') if l.startswith('rec')]; steps = [float(re.search(r'step\s+([+-]?[0-9.]+)', l).group(1)) for l in kinds]
assert len(kinds) == 13 and all(abs(s) < 0.5 for s in steps)                                           # the text says: no step is left at any of these places
app = [l for l in open(MV + 'page/apprun_b.log') if l.startswith('recording')]; assert len(app) == 29 and all('apart from y: 0.00 cents at most, followed or not differs on 0.000' in l for l in app)
same = [l for f in ('same0.log', 'same1.log') if os.path.exists(MV + 'same/' + f) for l in open(MV + 'same/' + f) if 'alone:' in l]; assert same and all(l.count('same') == 2 for l in same), 'something differs from build 0.4.7'
info = json.load(open(os.path.join(HERE, 'move050_info.json'))); nums = sorted(v['number'] for v in info['examples'].values()); first, last = nums[0], nums[-1]; assert nums == list(range(first, last + 1))
words = {8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}
OTHER = ('Loudness at far left brings a swell, a pass-by, a fade or steps of 4 to 6 dB down to 0.1 to 0.5 dB and leaves the single pulses of a low engine alone. Stereo at far left brings a pan sweep of 7 to 13 dB down to 0.3 dB or less, '
         'and the level changes by less than 0.5 dB. Tone reads a tilt or a sweep of 1.5 to 3.6 dB to within 0.3 to 0.8 dB (1.6 for a narrow bump that sweeps through a chord) and holds it about as still as a steady sound is by itself. Spectral at far left brings a colouring of 3 to 8 dB '
         'against neutral down to 0.5 to 1.8 dB, and at far right it brings a second sound to within 0.8 to 1.3 dB of the first sound\'s colour.')
h = []
h.append('<section id="move050"><h2>Movement: Pitch, Loudness, Tone, Stereo and Spectral. New in build 0.5.0</h2>')
h.append('<p>This is the group you asked for. MOVEMENT is always on the screen, before any found effect, with five faders: Pitch, Loudness, Tone, Stereo, and the Spectral fader you added. All five run the same way. Far left is flat: the movement is taken out. '
         'The middle is the sound\'s own, and there nothing is done at all: the sound comes out exactly as it went in. Far right with one sound is the sound\'s own movement twice. With two sounds the group works on the second sound, and far right puts the first sound\'s movement in the place of its own.</p>')
h.append('<p>Pitch is how the pitch moves: a rev, a glide, a pass-by, a vibrato. It is put on like tape speed. The sound is read at a moving place, nothing is stretched or built anew, so nothing warbles and nothing gets dull, and the sound keeps its length. '
         'Loudness is the level, read every 5 ms. Tone is how the balance of the sound\'s bands shifts over time, and Spectral is the colour the sound has on average. Stereo is where the sound sits between the sides and how wide it is.</p>')
h.append('<p>Pitch decided this build. You called the pitch changes of an earlier demo bad, most of all on the engine, so the reader of pitch movement has one rule above all others: where it cannot read, its curve stands still. It never fills a passage with a guess, '
         'and what it does not read stays in the sound. Holding that rule on your recordings took nine versions of the reader. The eighth still stepped away and back in 11 places of 29 recordings, by 90 cents and more, where it took a new sound to start and the sound in truth ran on. '
         'The ninth leaves no step at any of the %d places of that kind. The price is that it reads less: a short piece of sound beside a held one is now left alone.</p>' % len(kinds))
h.append('<p>The reader on test sounds whose true movement is known. The error is in cents, and 100 cents are a semitone.</p>')
h.append('<div class="tablewrap"><table><tr><th></th><th class="n">Followed</th><th class="n">Error in cents</th><th class="n">Movement read</th></tr>%s%s</table></div>' % (''.join(rows), none))
h.append('<p>What it does not read, said plainly. Of your 40 recordings the pitch is followed over four fifths of the sound or more in %d, over two fifths to four fifths in %d, over less than that in %d, and hardly at all in %d, which are noise-like. '
         'So on about half of your recordings the Pitch fader does little or nothing, and where it reads nothing at all it says so ("no pitch"). Left unread: noise; an engine whose partials lie closer than about 25 Hz while it revs fast; the very top of a fast rev and a drop from one moment to the next; '
         'a short sound beside a held one; of a sound that bounces up and down only a part; and where two things move at once, one of them or neither.</p>' % (n80, n40, n10, n0))
h.append('<p>The other four faders on test sounds with a known movement: %s</p>' % E(OTHER))
h.append('<p>Nothing else in the app has changed. With the MOVEMENT faders in the middle, %d test cases, two of every effect type, come out the same as in build 0.4.7 to the last bit, alone and on a second sound.</p>' % len(same) if len(same) == 38
         else '<p>Nothing else in the app has changed. With the MOVEMENT faders in the middle, %d test cases, one of every effect type, come out the same as in build 0.4.7 to the last bit, alone and on a second sound.</p>' % len(same))
h.append('<p>Examples %d to %d use your own recordings, picked by fixed rules from what the app reads in each of the 40, not by ear. For the three Pitch examples a picture shows the curve the app read. '
         'Examples %d and %d put the movement of example %d on a second sound, which in example %d is a sound from the test material.</p></section>' % (first, last, last - 1, last, first, last))
open(os.path.join(HERE, 'move050_head.html'), 'w').write('\n'.join(h) + '\n')
print('head written | recordings followed: %d / %d / %d / %d | places where another sound may start: %d, largest step %.1f | same as 0.4.7: %d cases' % (n80, n40, n10, n0, len(kinds), max(abs(s) for s in steps), len(same)))
