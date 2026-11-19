# The head of the page's 0.4.7 section (new047_head.html): the texts, with the table's numbers computed from the two
# scoreboard files (bench/famboard.py on all 136 cases: build 0.4.6 and build 0.4.7).
#   python3 new047_head.py [OLD.jsonl NEW.jsonl]
# Typed in from the engines' own notes (README.md in Research/sides, ring, crush and stutter): S below.
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = '/tmp/claude-0/v2/bench/'
OLD, NEW = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (BENCH + 'famboard_046b.jsonl', BENCH + 'famboard_047f.jsonl')
S = dict(haas_fresh=(56, 56), wide_fresh=(37, 42),                 # st7.py: 112 fresh cases; found of those that can be read at all
         ring_more=476, ring_missed=1, ring_hz=0.0014,             # rt7.py (80 fresh cases) and rk2.py (396: every clip with a tone put on, twice)
         crush_fresh=(56, 56), crush_filter=(1, 14),               # bf7.py: the four kinds it reads; the crusher with a filter after it
         stut_clips=198, stut_read=1152, stut_right=1138,          # sk2.py: every clip of the corpus with a stutter put on
         clean=198, others=384, forms=4)                           # no wrong find: clean clips, scoreboard sounds of the other families; pcm16.py: 16-bit and 24-bit forms
rows = lambda f: sorted((json.loads(l) for l in open(f) if l.strip()), key=lambda r: r['name'])
old, new = rows(OLD), rows(NEW); assert len(old) == 136 and len(new) == 136 and [r['name'] for r in old] == [r['name'] for r in new]
FIVE = [('haas', 'HAAS WIDTH'), ('width', 'STEREO WIDTH'), ('ringmod', 'RING MOD'), ('bitcrush', 'BITCRUSH'), ('stutter', 'STUTTER')]
r0 = lambda v: '%.0f' % (round(v) + 0.0)
def st(q, fam):
    r = [x for x in q if x['fam'] == fam]; assert len(r) == 8
    return dict(shown=sum(bool(x['found']) for x in r), out=None if any(x['out'] is None for x in r) else float(np.median([x['out'] for x in r])), onto=float(np.median([x['onto'] for x in r])),
                lo=min(x['onto'] for x in r if x['found']), hi=max(x['onto'] for x in r if x['found']), worse=sum(x['onto_all'] < 0 for x in r))
T = {f: (st(old, f), st(new, f)) for f, _ in FIVE}
for f, _ in FIVE: print(f, 'old', T[f][0], '\n      new', T[f][1])
five = [f for f, _ in FIVE]; by = {r['name']: r for r in new}; byo = {r['name']: r for r in old}
assert not [r['name'] for r in new if any(o in five for o in r['others'])]                    # the text says: none of the five groups on a case of another effect type
gone = [(r['name'], o) for r in old for o in r['others'] if o not in by[r['name']]['others'] and (r['fam'] in five or o in five)]; print('groups that are gone:', gone)
assert sorted(gone) == [('rng_mda_01', 'delay'), ('sat_swh_valve_01', 'stutter'), ('wid_calf_haasenh_01', 'comb')]
same = all(abs((a['out'] or 0) - (b['out'] or 0)) < 0.05 and abs(a['onto'] - b['onto']) < 0.05 and bool(a['found']) == bool(b['found']) for a, b in zip(old, new) if a['fam'] not in five); assert same      # 'the other twelve score exactly as before'
worse_old = sum(T[f][0]['worse'] for f in five); worse_new = sum(T[f][1]['worse'] for f in five); assert worse_new == 0
calf = sorted((by[n]['out'] for n in ('bit_calf_00', 'bit_calf_01')), reverse=True); assert all(byo[n]['out'] == 0 for n in ('bit_calf_00', 'bit_calf_01'))
mono = sorted(by[n]['onto'] for n in ('haas_mda_00', 'haas_ffmpeg_01')); wide = sorted(r['onto'] for r in new if r['fam'] == 'haas' and r['name'] not in ('haas_mda_00', 'haas_ffmpeg_01'))
filt = [by[n] for n in ('bit_mda_00', 'bit_mda_01')]; assert sum(bool(r['found']) for r in filt) == 1
cell = lambda x, y: '<td class="n">%s<br><small>%s</small></td>' % (x, y)
def out_cell(f):
    a, b = T[f]
    if f == 'stutter': return '<td class="n">cannot be</td>'
    if f == 'bitcrush': return '<td class="n">%s and %s<br><small>where it can be: 2 of 8</small></td>' % (r0(calf[0]), r0(calf[1]))
    return cell(r0(b['out']), r0(a['out']))
trs = ''.join('<tr><td>%s</td>%s%s%s%s</tr>' % (name, cell(T[f][1]['shown'], T[f][0]['shown']), out_cell(f), cell(r0(T[f][1]['onto']), r0(T[f][0]['onto'])), cell(T[f][1]['worse'], T[f][0]['worse'])) for f, name in FIVE)
html = ('<section id="new047"><h2>Stereo width, ring modulation, bitcrush and stutter: new in build 0.4.7</h2>'
 '<p>Five groups get an engine of their own in this build: HAAS WIDTH, STEREO WIDTH, RING MOD, BITCRUSH and STUTTER. Up to build 0.4.6 the app could tell that such an effect was on a sound and little more. '
 'What it then took out, or put on a second sound, was a rough stand-in for the effect, and on %d of the 40 test cases of these five types the second sound came out worse than untouched. '
 'Each of these effects does something exact to a sound, and the new engines read exactly that back: the few filter taps between the two sides, the pitch and phase of a ring modulator\'s tone, '
 'a crusher\'s grid of values and its hold, every repeat of a stutter to the sample.</p>'
 '<div class="tablewrap"><table><tr><th></th><th class="n">Shown, of 8</th><th class="n">Taken out: typical score</th><th class="n">On a second sound: typical score</th><th class="n">Second sound worse than untouched</th></tr>%s</table></div>'
 '<p>Eight test cases per type, each an outside effect (not my code) on a sound of 6 seconds. Large figures are this build, small ones build 0.4.6. The scores are of 100 as everywhere on this page, '
 'and the typical score is the middle of all eight, a case without its group counting 0. The other twelve effect types score exactly as in build 0.4.6, case by case.</p>'
 '<h2>What each engine does, and what it cannot do</h2>'
 '<p><b>HAAS WIDTH and STEREO WIDTH.</b> A Haas delay makes one side late. A widener sends the middle of the sound to the two sides through two short filters that differ, or puts a filtered copy of the middle at the sides. '
 'The engine finds those filters and works out the one sound behind both sides, and the Amount fader moves between that sound and the file as it came in. With a plain delay the late side moves in time on the way, so no comb is heard. '
 'On 112 further test sounds all %d Haas delays were found, and %d of the %d wideners that can be read at all. What cannot be read is a widener that simply makes the sides louder (two of the eight width cases here): nothing ties louder sides to the middle. '
 'The score for a Haas delay on a second sound says little. The outside tools fold the second sound to mono before they work and the app keeps what that sound has at its sides, so the score counts the difference: '
 'the two cases whose second sound is mono score %s and %s, the six with a wide one %s to %s.</p>'
 '<p><b>RING MOD.</b> A ring modulator multiplies the sound by a steady tone. The engine reads the tone\'s pitch, its phase and how much plain sound was left beside the modulated part, and settles them on the samples themselves. '
 'Amount takes the tone out again or deepens it, and Carrier moves its pitch an octave each way. On %d further test sounds it missed %s, and the pitch was never off by more than %.4f Hz. '
 'A tone whose pitch moves is not followed, a tone on only a part of the sound is not found, and under an echo or a reverb the tone is found about half the time.</p>'
 '<p><b>BITCRUSH.</b> A crusher rounds the samples to a grid of values and holds each value for several samples. Both are read exactly, and a second sound goes through the same crusher; Bit Depth and Sample Rate change its steps and its hold. '
 'Taking a crusher out is another matter: what the rounding and the hold threw away is gone, so on the sound itself the Amount fader does nothing left of the middle (example 37). '
 'One kind is the exception. A crusher that mixes plain sound back in loses nothing, and for it far left gives the sound before the crusher (two of the eight cases, example 36). '
 'Not read yet: a crusher with a filter after it. That is two of the eight cases; the group comes up on one of them, with its grid only.</p>'
 '<p><b>STUTTER.</b> A stutter plays a slice of the sound several times over, in place of what came after it. The engine finds every repeat to the sample: on the %d test sounds with a stutter of my own it read %s repeats, %s of them exactly right. '
 'On a second sound its own slices are played again at the same times, and Slice makes them shorter or longer. A stutter cannot be undone either, because what the repeats covered is not in the file. '
 'So far left plays each slice once and closes the sound up, and the sound ends early by what the repeats took (example 38). That is why the table has no score there: the score compares with the dry sound at the same times. '
 'A slice that is silent is not seen (example 39 has one).</p>'
 '<p><b>Files that are not at 48 kHz.</b> The app works at 48 kHz and converts other files on the way in. That conversion takes a crusher\'s values off their grid and makes a stutter\'s repeats unequal, '
 'so these two engines read such a file at its own rate (example 39).</p>'
 '<p><b>Effects that are not there.</b> None of the five groups comes up on the %d clean test sounds, nor on the %d test sounds of the other effect types, and a crusher or a stutter is not read into an ordinary 16-bit or 24-bit file. '
 'Three groups that build 0.4.6 showed without cause are gone: a DELAY on a ring modulation, a COMB FILTER on a widener and a STUTTER on a saturation. '
 'One limit to know: a 16-bit file that was made louder by 18 dB or more and saved as it is counts as a light crusher.</p>'
 '<p><b>Also mended.</b> A file without any sound in it is refused with a message, and values in a file that are not numbers count as silence.</p>'
 '<p>The seven examples below were picked by fixed rules, not by ear. For HAAS WIDTH, STEREO WIDTH and RING MOD: of the cases on which the group is shown, the middle one by its score for taking the effect out. '
 'For the crusher that can be undone: the weaker of its two cases. For the other crushers and for the stutter: the middle case by its score on the second sound. The last example is a file of my own making at 44.1 kHz.</p></section>'
 % (worse_old, trs, S['haas_fresh'][0], S['wide_fresh'][0], S['wide_fresh'][1], r0(mono[0]), r0(mono[1]), r0(wide[0]), r0(wide[-1]),
    S['ring_more'], {1: 'one'}[S['ring_missed']], S['ring_hz'], S['stut_clips'], '{:,}'.format(S['stut_read']), '{:,}'.format(S['stut_right']), S['clean'], S['others']))
open(os.path.join(HERE, 'new047_head.html'), 'w').write(html); print(html.replace('</p>', '</p>\n'))
