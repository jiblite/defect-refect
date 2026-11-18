# The head of the page's 0.4.4 section (new044_head.html): the texts, with the table's numbers computed from the two
# scoreboard result files.
#   python3 new044_head.py FAMBOARD_043.jsonl FAMBOARD_044.jsonl
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def fam(path):
    d = {}
    for r in map(json.loads, open(path)): d.setdefault(r['fam'], []).append(r)
    return d
def r0(v): return '%.0f' % (round(v) + 0.0)
def lst(v): v = [r0(x) for x in v]; return ', '.join(v[:-1]) + ' and ' + v[-1]
old, new = fam(sys.argv[1]), fam(sys.argv[2]); rows = []; med = lambda rs, k: float(np.median([r[k] for r in rs]))
for key, name in (('chorus', 'Chorus'), ('filtersweep', 'Filter sweep'), ('phaser', 'Phaser')):
    a, b = old[key], new[key]
    rows.append('<tr><td>%s</td><td class="n">%d of %d<br><small>%d</small></td><td class="n">%s<br><small>%s</small></td><td class="n">%s<br><small>%s</small></td></tr>'
                % (name, sum(r['found'] for r in b), len(b), sum(r['found'] for r in a), r0(med(b, 'out')), r0(med(a, 'out')), r0(med(b, 'onto')), r0(med(a, 'onto'))))
ch = sorted(new['chorus'], key=lambda r: -r['out']); good = [r for r in ch if r['out'] >= 50]; poor = [r for r in ch if r['out'] < 50]; cho_old = [r for r in old['chorus'] if r['found']]
sw = new['filtersweep']; ph = sorted([r for r in new['phaser'] if r['found']], key=lambda r: -r['out']); ph_old = [r for r in old['phaser'] if r['found']]
assert all(r['found'] for r in ch) and all(r['found'] for r in sw) and len(poor) == 3 and all(r['out'] < -100 for r in cho_old) and all(r['onto'] < -100 for r in ph_old) and all(r['out'] > 0 and r['onto'] > 0 for r in ph)
num = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
html = ('<section id="new044"><h2>Chorus, phaser and filter sweep: new in build 0.4.4</h2>'
 '<p>CHORUS has a new engine. A chorus is the sound plus one or more copies of itself, each a few milliseconds behind it, with a delay that drifts back and forth. The engine finds each copy, follows its delay through the sound and reads how loud it is. '
 'Taking the chorus out is the exact opposite filter, and the same copies go onto a second sound. Speed makes the drift slower or faster. Depth makes it narrower or wider, and at far left the copies stand still. '
 'A copy that is louder than the sound itself is told from the sound and taken out as well (example 19).</p>'
 '<p>PHASER and FILTER SWEEP share one new engine of another kind. Both are a tone curve that moves: a filter that opens and closes, or notches that travel up and down. '
 'A small learned model looks at the picture of the sound, pitch against time, and draws the curve it believes was put on it, moment by moment. Taking the effect out is that curve upside down, and putting it on a second sound is the same curve on that sound. Speed plays the curve slower or faster. '
 'The model learned from 16,000 examples of such effects on other sounds. Half of the effects it is tested on below, two of the four sweeps and two of the four phasers, are tools it never saw.</p>'
 '<p>All three work on long sounds and say where along the sound the effect is. A filter sweep or phaser that is on a part of the sound only is taken out of that part, and the rest comes back as it went in.</p>'
 '<p>The test cases of 6 seconds: eight per effect, made with effects that are not my code, each scored against the true answer (100 = identical to it, 0 = no better than leaving the sound alone). '
 'Large figures are this build, small ones build 0.4.3; a case that is not shown counts 0.</p>'
 '<div class="tablewrap"><table><tr><th></th><th class="n">Shown</th><th class="n">Taken out</th><th class="n">Onto a second sound</th></tr>%s</table></div>'
 '<p>Chorus: %s of the eight come out at %s to %s. The other three are found but hardly taken out (%s): in two of them the copies drift at different speeds and the engine reads only one speed, and in the third the copy is fed back into itself. Onto a second sound those three still land at %s to %s. '
 'Build 0.4.3 showed %s of the eight and made both far worse than they went in.</p>'
 '<p>Filter sweep: all eight are found and come out at %s to %s. The curve is found in the right place but read too small, so a part of the sweep stays in. Onto a second sound that matters less: %s to %s.</p>'
 '<p>Phaser: the %s that are shown come out at %s and go onto a second sound at %s. Build 0.4.3 showed %s and both went onto a second sound far worse than nothing.</p>'
 '<p>On 198 clean test sounds none of the three groups is shown. On about 600 sounds with other effects on them CHORUS and FILTER SWEEP are never shown, and PHASER once, on a reverb whose tail moves.</p>'
 '<p>Long sounds, 30 to 48 seconds. On 12 clean ones nothing is shown. A filter sweep on all of the sound is shown on 7 of 8 and comes out at 19 to 32. One that comes in half way is shown on 4 of 4, and what lies before it comes back exactly as it went in on 3 of the 4 (on the fourth the app has the filter start 2.6 seconds early). '
 'A chorus on a 42-second sound comes out at 81 and goes onto a second sound at 91 (example 20). A phaser on a long sound is shown on 1 of 8.</p>'
 '<p>What is not good yet. The phaser is the weak one. It is shown only when the model is very sure, on 7 of the 16 test sounds, and is then taken out by about a quarter. A light phaser is not told apart from the sound\'s own movement, and fitting the model for longer did not change that. The next step there is a larger model, or a reader that uses how a phaser\'s notches keep their shape while they travel. '
 'A gentle filter sweep, under about 8 dB, is not shown at the default Sureness. A chorus whose copy is about as loud as the sound comes out only partly, a faint one can be missed, and reading a chorus takes about half the sound\'s length. '
 'And on some chorus, phaser and sweep sounds the app still shows REVERB, which is not there.</p></section>'
 % (''.join(rows), num[len(good)], r0(good[-1]['out']), r0(good[0]['out']), lst([r['out'] for r in poor]), r0(min(r['onto'] for r in poor)), r0(max(r['onto'] for r in poor)), num[len(cho_old)],
    r0(min(r['out'] for r in sw)), r0(max(r['out'] for r in sw)), r0(min(r['onto'] for r in sw)), r0(max(r['onto'] for r in sw)),
    num[len(ph)], lst([r['out'] for r in ph]), lst([r['onto'] for r in ph]), num[len(ph_old)]))
open(os.path.join(HERE, 'new044_head.html'), 'w').write(html); print(html.replace('</p>', '</p>\n'))
