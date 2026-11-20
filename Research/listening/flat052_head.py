# The head of the page's 0.5.2 section (flat052_head.html): the texts, with every number computed from result files:
# the trial (Research/declip/trial.py: the app's code against build 0.5.1's way back, 84 outside cases and 162 soft
# clippings), the two scoreboard files (bench/famboard.py on all 136 cases: build 0.5.1 and build 0.5.2), the twelve
# hard clippers' cases with a second sound (Research/declip/second.py), and the examples' own numbers (flat052_info.json).
#   python3 flat052_head.py
# Typed in from the engine's notes (Research/declip/README.md): S below.
import os, json, glob, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = '/tmp/claude-0/v2/declip/data/'; TRIAL = os.environ.get('TRIAL', 'r3')
BOARD = os.environ.get('BOARD', '/tmp/claude-0/v2/declip/fb052.jsonl'); BOARD_OLD = os.environ.get('BOARD_OLD', '/tmp/claude-0/v2/declip/fb051.jsonl'); SECOND = os.environ.get('SECOND', '/tmp/claude-0/v2/declip/second12.json')
S = dict(clean=198, clean_shown=4,                       # Research/declip/clean.py: clean test sounds on which DISTORTION or SATURATION is shown, as in build 0.5.1
         full=(44, 29, 43),                              # Research/declip/README.md, the level: a cut file brought to full scale, as it was meant / with the limiter on the rebuilt tops / now
         long_s=48, long_wait=12, cut_wait='a tenth',    # Research/declip/README.md: a soft top on a 48 second file; a plain cut
         over=((6, 9), (23, 15, 9), (14.5, 6, 3)))       # second.py: the meter on a plain cut of a dense sound (dB too deep; read; true)
r0 = lambda v: ('%.0f' % (round(v) + 0.0)).replace('-', '−')            # whole numbers, with a real minus sign
n_word = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}
def trial(part):
    rows = []
    for f in sorted(glob.glob(DATA + 'd38_%s_%s_?.pkl' % (TRIAL, part))): rows += pickle.load(open(f, 'rb'))
    return rows
A = [r for r in trial('all') if not r['name'].startswith('x_')]; M = trial('many'); assert len(A) == 84 and len(M) == 162
mean = lambda R, k: float(np.mean([r[k] for r in R])); under = lambda R, k: sum(r[k] < -0.05 for r in R)

# ---------------------------------------------------------------- the 84 outside cases, by what the engine found at the top
soft = [r for r in A if 'soft' in r.get('kinds', '')]; hard = [r for r in A if 'hard' in r.get('kinds', '') and r not in soft]; none = [r for r in A if r not in soft and r not in hard]
assert len(soft) == 15 and len(hard) == 13 and len(none) == 56
assert all(abs(r['new'] - r['plain']) < 1e-6 for r in none)                                    # 'to the last bit'
assert min(r['new'] - r['plain'] for r in A) > -0.05                                           # 'no case worse than in build 0.5.1'
hard_reb = [r for r in hard if r['rebuilt']]; hard_ref = [r for r in hard if not r['rebuilt']]; assert len(hard_reb) == 7 and len(hard_ref) == 6 and all(abs(r['new']) < 1e-6 for r in hard_ref)
clip12 = [r for r in A if r['name'].startswith(('clip_clip_sox_', 'clip_clip_pedalboard_', 'clip_clip_ff_hard_'))]; assert len(clip12) == 12
print('84: soft %d: %.1f -> %.1f (worse than left alone %d -> %d) | plain cut %d: %.1f -> %.1f (%d -> %d), rebuilt %d refused %d | neither %d: %.2f | all: %.1f -> %.1f (%d -> %d) | the 12 hard clippers in 0.5.1: %.1f'
      % (len(soft), mean(soft, 'plain'), mean(soft, 'new'), under(soft, 'plain'), under(soft, 'new'), len(hard), mean(hard, 'plain'), mean(hard, 'new'), under(hard, 'plain'), under(hard, 'new'), len(hard_reb), len(hard_ref),
         len(none), mean(none, 'new'), mean(A, 'plain'), mean(A, 'new'), under(A, 'plain'), under(A, 'new'), mean(clip12, 'plain')))

# ---------------------------------------------------------------- the 162 soft clippings
reb = [r for r in M if r.get('rebuilt')]; rest = [r for r in M if not r.get('rebuilt')]; gain = [r['new'] - r['plain'] for r in reb]
up = sum(g >= 5 for g in gain); down = sum(g <= -5 for g in gain); assert all(abs(r['new'] - r['plain']) < 1e-6 for r in rest if not r.get('did'))
print('162: rebuilt %d, mean gain %.1f, better by 5+ %d, worse by 5+ %d | rebuilt: %.1f -> %.1f, worse than left alone %d -> %d | not rebuilt %d' % (len(reb), np.mean(gain), up, down, mean(reb, 'plain'), mean(reb, 'new'), under(reb, 'plain'), under(reb, 'new'), len(rest)))

# ---------------------------------------------------------------- the scoreboard
rows = lambda f: {r['name']: r for r in (json.loads(l) for l in open(f) if l.strip())}
new, old = rows(BOARD), rows(BOARD_OLD); assert len(new) == 136 and sorted(new) == sorted(old)
same = lambda a, b: abs((a['out'] or 0) - (b['out'] or 0)) < 0.1 and abs(a['onto'] - b['onto']) < 0.1 and bool(a['found']) == bool(b['found']) and abs((a['out_all'] or 0) - (b['out_all'] or 0)) < 0.1 and abs((a['onto_all'] or 0) - (b['onto_all'] or 0)) < 0.1 and sorted(a['others']) == sorted(b['others']) and sorted(a.get('shown', [])) == sorted(b.get('shown', []))
changed = sorted(n for n in new if not same(new[n], old[n])); assert changed == ['dist_pedalboard_00', 'dist_pedalboard_01'], changed          # to a tenth of a point (the level rule moves one distortion case by 0.07)
c0, c1 = [(old[n], new[n]) for n in changed]; assert all(abs(a['onto'] - b['onto']) < 0.05 for a, b in (c0, c1))            # their second sounds as before
dist = sorted(n for n in new if new[n]['fam'] in ('distortion', 'saturation') and 'distortion' in new[n].get('shown', [])); assert len(dist) == 7 and dist == sorted(n for n in old if old[n]['fam'] in ('distortion', 'saturation') and 'distortion' in old[n].get('shown', []))
d_new = [new[n]['out'] for n in dist]; d_old = [old[n]['out'] for n in dist]; d_gain = sum(v > 0 for v in d_new); d_under = sorted((v for v in d_new if v < 0), reverse=True)
print('scoreboard: changed', [(a['name'], round(a['out'], 1), round(b['out'], 1)) for a, b in (c0, c1)], '| the %d shown distortions taken out: %.1f -> %.1f' % (len(dist), np.mean(d_old), np.mean(d_new)), [round(v, 1) for v in d_new])

# ---------------------------------------------------------------- a cut on a second sound
two = json.load(open(SECOND)); assert len(two) == 12
t_reb = [v for v in two.values() if v['new']['rebuilt']]; t_not = [v for v in two.values() if not v['new']['rebuilt']]; assert len(t_reb) == 6 and len(t_not) == 6
assert all(abs(v['new']['onto'] - v['old']['onto']) < 0.05 for v in t_not) and all(v['new']['onto'] > v['old']['onto'] + 5 for v in t_reb)      # 'exactly as in build 0.5.1'; all six better
far = sorted(v['new']['onto'] for v in t_not if v['new']['onto'] < -100); assert len(far) == 3
reb_new = sorted(v['new']['onto'] for v in t_reb); print('second sound: rebuilt six %.1f (0.5.1 %.1f), from %.1f to %.1f | not rebuilt, far worse than untouched:' % (np.mean(reb_new), np.mean([v['old']['onto'] for v in t_reb]), reb_new[0], reb_new[-1]), [round(v) for v in far])
assert all(v['new']['over'] <= 1.0 + 1e-9 for v in two.values())                                # 'no value comes out higher than the file's highest was'

# ---------------------------------------------------------------- the examples
info = json.load(open(os.path.join(HERE, 'flat052_info.json'))); tags = list(info); assert tags == ['c', 's1', 's2', 'a', 'w']
FIRST = int(os.environ.get('FIRST', '52')); ex = {t: FIRST + i for i, t in enumerate(tags)}
assert info['s1']['case'] == changed[0] and info['s2']['case'] == changed[1] and info['w']['over'] in (2, 4) and info['w']['scores']['out'] < info['w']['scores']['out51']

# ---------------------------------------------------------------- the texts
cell = lambda x, y: '<td class="n">%s<br><small>%s</small></td>' % (x, y)
def tr(name, R, note=''): return '<tr><td>%s%s</td><td class="n">%d</td>%s%s</tr>' % (name, note, len(R), cell(r0(mean(R, 'new')), r0(mean(R, 'plain'))), cell(under(R, 'new'), under(R, 'plain')))
table = ('<div class="tablewrap"><table><tr><th></th><th class="n">Sounds</th><th class="n">Taken out: average score</th><th class="n">Worse than left alone</th></tr>%s%s%s%s</table></div>'
         % (tr('Tops pressed flat by a curve', soft), tr('Tops cut off', hard), tr('No flat top', none), tr('162 harder soft clippings: the %d that are rebuilt' % len(reb), reb)))
html = ('<section id="flat052"><h2>Flat tops rebuilt: new in build 0.5.2</h2>'
 '<p>This is the declipper you picked. Build 0.5.1 could read the curve a sound was pushed through and walk it back, but it said of itself that tops which were pressed flat stay flat: '
 'where a curve pressed many levels into one, nothing in a value says how high it stood. The sound around a flat top does say it. '
 'The new engine makes up what each top was from the sound before and after it, and the far left of DISTORTION, and of SATURATION, now gives the rebuilt sound. Nothing was added to the screen: the same group, the same fader.</p>'
 '<p>The app looks at each half of the wave by itself and tells three things apart. <b>Tops cut off</b>: many values sit at exactly one level, as after a clipper or an overloaded input. '
 'Nothing under the cut is touched, and the values at the cut are rebuilt above it. This is looked for on every sound, and such a sound shows DISTORTION whatever the old reading says. '
 '<b>Tops pressed flat by a curve</b>: the values pile up just under the top, on a sound where a bend was found. The curve is walked back as far as it can be read and the tops are rebuilt. '
 'The rebuilt tops then tell how the curve runs further up, and with the whole curve known the tops are rebuilt a second time. <b>Neither</b>: the sound goes the way of build 0.5.1, to the last bit.</p>'
 '<p>A rebuilt top is made up, so the engine checks itself before it lets one out. It first tries its hand on the sound in front of it: it cuts tops that are whole, rebuilds them and compares. '
 'A dense, noise-like sound fails that test, because nothing in it says what its tops were. Such a sound is left as it is: left of the middle the fader then does nothing, and the fader\'s tip says so. '
 'Four more checks stop a rebuild that would do harm: a sound cut almost to a square wave is left alone, both halves of a wave are taken the same way, a first pass that contradicts itself is thrown out, and so is a second pass that leaves the tops rough. The group\'s picture says in words what was found on the sound in hand, and how high the rebuilt tops stand.</p>'
 '%s'
 '<p>The first three rows are %d test sounds through outside distortions, saturations and clippers (not my code), sorted by what the app finds at their tops. Large figures are this build, small ones build 0.5.1; '
 'the scores are of 100 as everywhere on this page, for Amount at far left. Of the %d sounds with cut-off tops %s are rebuilt and %s fail the self-test and are left alone, where build 0.5.1 made %s of those %s worse. '
 'No sound of the %d comes out worse than in build 0.5.1. The last row is a second trial made to be hard: 144 sounds through soft clippers of eight kinds, a third of them working at twice and a third at four times the sample rate inside, which leaves ripple on the tops, and 18 through a plain distortion. '
 'Of its %d sounds %d are rebuilt: %d of them gain 5 points or more, %d lose 5 or more, and the average gain is %s. The other %d are not rebuilt and come out as in build 0.5.1.</p>'
 % (table, len(A), len(hard), n_word[len(hard_reb)], n_word[len(hard_ref)], n_word[under(hard_ref, 'plain')], n_word[len(hard_ref)], len(A), len(M), len(reb), up, down, r0(np.mean(gain)), len(rest)))
html += ('<p><b>The scoreboard.</b> Of the 136 cases two change, the two distortions whose tops are pressed flat: they are examples %d and %d below. Taken out they now score %s and %s, where build 0.5.1 scored %s and %s. '
 'The other 134 cases score as they did in build 0.5.1. Over the %s distortions that the app shows on the scoreboard, taking the effect out now scores %s on average (%s before): %s with a gain and %s a little worse than left alone (%s). '
 'Clean sounds are not taken for clipped more often than before: of %d clean test sounds %s show DISTORTION or SATURATION, the same %s as in build 0.5.1.</p>'
 % (ex['s1'], ex['s2'], r0(c0[1]['out']), r0(c1[1]['out']), r0(c0[0]['out']), r0(c1[0]['out']), n_word[len(dist)], r0(np.mean(d_new)), r0(np.mean(d_old)), n_word[d_gain], n_word[len(d_under)], ' and '.join(r0(v) for v in d_under), S['clean'], n_word[S['clean_shown']], n_word[S['clean_shown']]))
assert d_gain + len(d_under) == len(dist) and len(d_under) == 2 and min(d_under) > -15            # 'a little worse'
html += ('<p><b>The level.</b> The rebuilt sound keeps the file\'s level: no value comes out higher than the file\'s highest was. Most clipped files are cut at full scale, and their tops must not come back over it. '
 'So the rebuilt sound is quieter than the clipped one, by about as much as its tops were cut.</p>'
 '<p><b>On a second sound.</b> A curve with pressed tops goes onto a second sound as in build 0.5.1. A cut whose tops were rebuilt goes on as a cut, and the second sound is pushed as far past it as the first sound\'s rebuilt tops stand. '
 'On the %s test cases of that kind the second sound scores %s on average (between %s and %s), where build 0.5.1 scored %s.</p>'
 % (n_word[len(t_reb)], r0(np.mean(reb_new)), r0(reb_new[0]), r0(reb_new[-1]), r0(np.mean([v['old']['onto'] for v in t_reb]))))
html += ('<h2>What the declipper cannot do</h2>'
 '<p><b>Most distortions and saturations are not rebuilt.</b> %d of the %d test sounds come out exactly as in build 0.5.1. Their effect is more than a curve, with a filter before or behind it or with plain sound mixed in, or it bends softly, and leaves no flat top to rebuild. Example 50 above is one of them. '
 '<b>Dense sounds are left alone</b> (example %d), and so is a sound that was clipped almost to a square wave. '
 '<b>A wave that was flat-topped before it was clipped</b> can come back too tall. One such case came up in the trials and is caught by a check, but the kind cannot be told from the sound. '
 '<b>A clipper that works at a higher sample rate inside</b> is rebuilt less often and gains less; example %d is one, and the case of the trial that loses most. '
 '<b>A sound that was bent only lightly</b> can come out a little under the old way. '
 'A long file is judged by its three most clipped seconds, and a file of more than 20 minutes is not rebuilt. Cut levels that differ between left and right are not handled. '
 'A few clipped peaks, with less than half a dB to rebuild, do not show DISTORTION by themselves.</p>'
 '<p><b>A cut whose tops are not rebuilt, on a second sound.</b> How hard the first sound was driven is then not known. The second sound gets the curve as build 0.5.1 read it, and on a dense sound that reading is %d to %d dB too deep. '
 'On %s of the %s test cases of that kind the second sound comes out far worse than untouched, exactly as it did in build 0.5.1. This build does not mend that.</p>'
 '<p><b>Waiting.</b> Rebuilding takes time when a file is dropped: for a plain cut about %s of the sound\'s length, for pressed tops about a quarter of a long sound\'s length (a %d-second file: about %d seconds) and up to all of a short one\'s. '
 'The thresholds of all the checks rest on about 250 test sounds, which is not many.</p>'
 % (len(none), len(A), ex['a'], ex['w'], S['over'][0][0], S['over'][0][1], n_word[len(far)], n_word[len(t_not)], S['cut_wait'], S['long_s'], S['long_wait']))
html += ('<p>The five examples below were picked by fixed rules, not by ear. Example %d: of the sounds through outside clippers whose tops are rebuilt, the middle one by score. Examples %d and %d: the two scoreboard cases that this build changes. '
 'Example %d: of the cut sounds that fail the self-test, the one build 0.5.1 made worst. Example %d: of the %d rebuilt sounds of the hard trial, the one that loses most against build 0.5.1. '
 'Each has the far left of build 0.5.1 next to the far left of this build.</p></section>' % (ex['c'], ex['s1'], ex['s2'], ex['a'], ex['w'], len(reb)))
if __name__ == '__main__': open(os.path.join(HERE, 'flat052_head.html'), 'w').write(html); print('flat052_head.html written,', len(html), 'characters')
