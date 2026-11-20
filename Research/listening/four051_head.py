# The head of the page's 0.5.1 section (four051_head.html): the texts, with the table's numbers computed from the two
# scoreboard files (bench/famboard.py on all 136 cases: build 0.5.0, which scores as 0.4.7 did, and build 0.5.1).
#   python3 four051_head.py [OLD.jsonl NEW.jsonl]
# Typed in from the engines' own notes (README.md in Research/filter, bend and exciter): S below.
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = '/tmp/claude-0/v2/bench/'
OLD, NEW = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (BENCH + 'famboard_047f.jsonl', BENCH + 'famboard_051.jsonl')
S = dict(fil_cases=32, fil_shown=22, fil_low=28, fil_out=50, fil_out_worse=0, fil_onto=63, fil_onto_worse=1, fil_off='0.6',     # Research/filter, fgrid.py: 32 outside cases, shown at the default / with the slider down, scores over the shown
         fil_missed=10, fil_never=4, clean=198, fil_clean=0, fil_others=624, fil_false=9, fil_faint=4,                           # fgrid.py, fothers.py; both.py: fixed filters on which the moving-tone engine suspects a sweep
         bend_sounds=96, bend_boxes=8, bend_off='1.8', bend_name=85, dist=(45, 48), sat6=(21, 25), soft=(3, 23),                 # Research/bend, beval.py: blind on the outside cases
         bend_clean=4, bend_others=(12, 608), dist_out=20,                                                                       # beval.py: clean corpus, the other effects' sounds; taking out where DISTORTION is shown (24 cases)
         exc_found=(8, 40), exc_before=(2, 40))                                                                                  # Research/exciter, eeval.py
rows = lambda f: sorted((json.loads(l) for l in open(f) if l.strip()), key=lambda r: r['name'])
old, new = rows(OLD), rows(NEW); assert len(old) == 136 and len(new) == 136 and [r['name'] for r in old] == [r['name'] for r in new]
THREE = [('filter', 'FILTER'), ('distortion', 'DISTORTION'), ('saturation', 'SATURATION')]; three = [f for f, _ in THREE]
r0 = lambda v: ('%.0f' % (round(v) + 0.0)).replace('-', '\u2212')            # whole numbers, with a real minus sign
def st(q, fam):
    r = [x for x in q if x['fam'] == fam]; assert len(r) == 8
    return dict(shown=sum(bool(x['found']) for x in r), out=float(np.median([x['out'] for x in r])), onto=float(np.median([x['onto'] for x in r])), worse=sum(x['onto_all'] < 0 for x in r), own_worse=sum(x['onto'] < 0 for x in r),
                out_worse=sum(x['out'] < 0 for x in r), ontos=sorted(x['onto'] for x in r if x['found']), outs=sorted(x['out'] for x in r if x['found']))
T = {f: (st(old, f), st(new, f)) for f in three + ['exciter']}
for f in T: print(f, 'old', T[f][0], '\n      new', T[f][1])
by = {r['name']: r for r in new}; byo = {r['name']: r for r in old}
NEWG = ('filter', 'distortion', 'saturation')
# the new engines' groups on cases of other effect types (a distortion on a saturation case and the other way round is the same engine's group)
stray = sorted((r['name'], o) for r in new for o in r['others'] if o in NEWG); stray_old = sorted((r['name'], o) for r in old for o in r['others'] if o in NEWG + ('exciter',)); print('new engines\' groups on other types\' cases:', stray, '| in 0.5.0 (with EXCITER):', stray_old)
rest = [f for f in sorted({r['fam'] for r in new}) if f not in three + ['exciter']]; assert len(rest) == 13
moved = [(a['name'], a['found'], b['found'], a['out'], b['out'], round(a['onto'], 1), round(b['onto'], 1), a['others'], b['others']) for a, b in zip(old, new) if a['fam'] in rest
         and not (abs((a['out'] or 0) - (b['out'] or 0)) < 0.05 and abs(a['onto'] - b['onto']) < 0.05 and bool(a['found']) == bool(b['found']) and abs(a['onto_all'] - b['onto_all']) < 0.05 and sorted(a['others']) == sorted(b['others']))]
print('cases of the other 13 types that changed:', moved)
exc_old = [r for r in old if r['fam'] == 'exciter' and r['found']]; assert len(exc_old) == 1 and exc_old[0]['onto'] < -100 and T['exciter'][1]['shown'] == 0
cell = lambda x, y: '<td class="n">%s<br><small>%s</small></td>' % (x, y)
trs = ''.join('<tr><td>%s</td>%s%s%s%s</tr>' % (name, cell(T[f][1]['shown'], T[f][0]['shown']), cell(r0(T[f][1]['out']), r0(T[f][0]['out'])), cell(r0(T[f][1]['onto']), r0(T[f][0]['onto'])), cell(T[f][1]['own_worse'], T[f][0]['own_worse'])) for f, name in THREE)
sat = T['saturation'][1]; dis = T['distortion'][1]; fil = T['filter'][1]
if __name__ == '__main__': print('saturation shown: onto', sat['ontos'], 'out', sat['outs'], '| distortion: onto', dis['ontos'], 'out', dis['outs'], '| filter: onto', fil['ontos'], 'out', fil['outs'])

# ---------------------------------------------------------------- what the text says, checked against the scoreboard
assert fil['shown'] >= 5 and fil['out_worse'] == 0 and fil['own_worse'] == 0                       # 'no case worse' for the filter, taken out and on a second sound
assert T['saturation'][0]['shown'] == 0 and T['filter'][0]['shown'] == 1 and T['distortion'][0]['shown'] == 4
bend_new = [r for r in new if r['fam'] in ('distortion', 'saturation')]; as_sat = sorted(r['name'] for r in bend_new if 'saturation' in r['shown']); as_dist = sorted(r['name'] for r in bend_new if 'distortion' in r['shown'])
print('shown as SATURATION:', as_sat, '| as DISTORTION:', as_dist)
dist_rows = [by[n] for n in as_dist]; sat_rows = [by[n] for n in as_sat]
d_onto = sorted(r['onto'] for r in dist_rows); d_out = sorted(r['out'] for r in dist_rows); s_onto = sorted(r['onto'] for r in sat_rows); s_out = sorted(r['out'] for r in sat_rows)
d_worse = [r['name'] for r in dist_rows if r['onto'] < 0]; d_out_worse = [r['name'] for r in dist_rows if r['out'] < 0]
# second sounds that come out worse than untouched with everything the app shows, on the cases of the three types, and what does it
worse_all = [(r['name'], round(r['onto_all']), round(r['onto']), r['others']) for r in new if r['fam'] in three and r['onto_all'] < 0]; print('second sound worse with everything shown:', worse_all)
by_reverb = [w for w in worse_all if w[2] >= 0 and 'reverb' in w[3]]; by_own = [w for w in worse_all if w[2] < 0]; assert len(by_reverb) + len(by_own) == len(worse_all)
worse_old = [(r['name'], round(r['onto_all'])) for r in old if r['fam'] in three + ['exciter'] and r['onto_all'] < 0]; print('the same in 0.5.0 (with the exciter cases):', worse_old)
worse_old3 = [r['name'] for r in old if r['fam'] in three and r['onto_all'] < 0]; assert len(worse_old3) > len(worse_all)

# the scoreboard's own distortion cases on which DISTORTION is shown
dist_own = [r for r in dist_rows if r['fam'] == 'distortion']; d_good = sorted(r['onto'] for r in dist_own if r['onto'] >= 0); d_bad = [r for r in dist_own if r['onto'] < 0]; d_out_bad = sorted((r['out'] for r in dist_own if r['out'] < 0), reverse=True)
assert len(dist_own) == len(dist_rows) and len(d_bad) == 1 and d_bad[0]['name'].startswith('dist_caps') and len(d_out_bad) >= 2 and min(d_out_bad) > -15     # 'one ... a saturation plugin driven hard', 'a little worse'
d_bad_txt = 'one comes out a little worse than untouched (%s): a saturation plugin driven hard, whose effect is more than a curve' % r0(d_bad[0]['onto'])
# ---------------------------------------------------------------- the texts
sec = lambda v: ', '.join(r0(x) for x in v[:-1]) + ' and ' + r0(v[-1])
n_word = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
assert len(as_sat) >= 2 and len(as_dist) >= 2 and not [r for r in sat_rows if r['onto'] < 0]            # 'none of the saturations worse on a second sound'
worse_txt = ('With everything the app shows on these 24 cases, the second sound comes out worse than untouched on %s (%s in build 0.5.0). On %s of them a REVERB group that is not there does it, as it did before; on %s the new group itself does (a distortion, see below).'
             % (n_word[len(worse_all)], n_word[len(worse_old3)], n_word[len(by_reverb)], n_word[len(by_own)])) if by_own else ('With everything the app shows on these 24 cases, the second sound comes out worse than untouched on %s, and each time a REVERB group that is not there does it, as in build 0.5.0.' % n_word[len(worse_all)])
html = ('<section id="four051"><h2>Fixed filter, distortion and saturation: new in build 0.5.1</h2>'
 '<p>Three groups get an engine of their own in this build: FILTER, for a low-pass, a high-pass or a band-pass that stands still while the sound plays, and DISTORTION and SATURATION. A fourth, EXCITER, was tested and is taken out of the app. '
 'Up to build 0.5.0 the app could tell that a filter or a distortion was probably on a sound and little more: what it then took out, or put on a second sound, was a stand-in with settings of its own making. '
 'The new engines read the effect itself off the sound: a filter\'s curve in 104 bands, and for distortion and saturation the curve that every value of the sound was pushed through.</p>'
 '<div class="tablewrap"><table><tr><th></th><th class="n">Shown, of 8</th><th class="n">Taken out: typical score</th><th class="n">On a second sound: typical score</th><th class="n">Second sound worse than untouched</th></tr>%s</table></div>'
 '<p>Eight test cases per type, each an outside effect (not my code) on a sound of 6 seconds. Large figures are this build, small ones build 0.5.0. The scores are of 100 as everywhere on this page, '
 'the typical score is the middle of all eight, a case without its group counting 0, and the last column counts the group by itself. %s</p>'
 % (trs, worse_txt))
sat_not = sorted(r['name'] for r in bend_new if r['fam'] == 'saturation' and not r['found'])
html += ('<h2>What each engine does, and what it cannot do</h2>'
 '<p><b>FILTER.</b> A filter leaves no mark that a sound could not have by itself: a sound without highs may be a filtered sound or a dull one. What the engine goes by is that sounds are not built the way filters are. '
 'A filter\'s skirt falls by the same number of dB in every frame, loud or quiet, and a sound\'s own spectrum moves from frame to frame. Two small readers each give the filter\'s curve in 104 bands, '
 'and the engine takes the smaller of the two readings in every band, because a cut that is read too deep does harm when it is undone. The group is named after what was found: LOW-PASS FILTER, HIGH-PASS FILTER or BAND-PASS FILTER. '
 'Amount takes the filter out, by lifting what it cut by up to 40 dB, or puts it on a second time. Cutoff moves the curve two octaves each way, and Resonance, there when the curve has a peak, scales the peak. '
 'The picture shows the curve as read, and with a fader moved the curve that results.</p>'
 '<p>On %d test cases from four outside filters, the eight of the table and 24 more, the group is shown on %d where the Sureness slider stands at first and on %d with the slider down. '
 'Over the %d, taking the filter out scores %d on average and makes no case worse than leaving it alone. Putting it on a second sound scores %d, and one case comes out worse than untouched: a high-pass whose corner is read %s of an octave too high.</p>'
 '<p>What it cannot do. Beyond a point a filtered sound is not told from one that is dull or thin by itself, and that goes both ways. A filter that is there is missed, or its corner is read too high or too low, where the sound has little near the corner: '
 '%d of the %d cases are not shown at first, %d not at all. A filter that is not there is shown on none of the %d clean test sounds and on %d of the %d test sounds of the other effect types, mostly a high-pass on a sound that has little low end. '
 'Where the filter left nothing of the sound, its curve is carried on and not read. Not read at all: a shelf or a bell of an equaliser, a notch, a filter on one side only, and a sound of under a second. '
 'A filter that moves stays with the FILTER SWEEP group.</p>'
 '<p><b>DISTORTION and SATURATION.</b> The two are one thing at two strengths: a fixed curve that every value of the sound is pushed through. Small values pass in a line, large ones are squeezed. '
 'The engine reads that curve off the sound, for the values above zero and for those below, because a tube or a tape does not bend the two alike. It can do that because a curve leaves the quiet frames of a sound as they were and presses the loud ones together: '
 'the same sound shows its own shape in the quiet frames and the bent shape in the loud ones. The group is called DISTORTION when the loudest values were squeezed by 10 dB or more, and SATURATION under that. '
 'It has one fader. On a second sound, Amount pushes that sound as far into the curve as the first sound was pushed: less far to the left, further to the right. On the sound itself, right pushes it further into the same curve and left walks the bend back. '
 'The picture shows the curve, what goes in against what comes out.</p>'
 '<p>Blind on %d sounds through %s outside effects, four that distort and four that saturate, the squeeze is read %s dB off on average and the name is right on %d. Shown where the slider stands at first: '
 '%d of the %d distorted sounds, %d of the %d that are saturated by 6 dB or more, and %d of the %d that are saturated by less. On the scoreboard\'s eight saturation cases that is %s: they go onto a second sound at %s. Of its eight distortion cases %s are shown. On a second sound %s of them score between %s and %s, and %s. Taken out, %s of them come out a little worse than left alone (%s): that is the first limit below.</p>'
 % (S['fil_cases'], S['fil_shown'], S['fil_low'], S['fil_shown'], S['fil_out'], S['fil_onto'], S['fil_off'],
    S['fil_missed'], S['fil_cases'], S['fil_never'], S['clean'], S['fil_false'], S['fil_others'],
    S['bend_sounds'], n_word[S['bend_boxes']], S['bend_off'], S['bend_name'], S['dist'][0], S['dist'][1], S['sat6'][0], S['sat6'][1], S['soft'][0], S['soft'][1],
    n_word[len([r for r in sat_rows if r['fam'] == 'saturation'])], sec(sorted(r['onto'] for r in sat_rows if r['fam'] == 'saturation')),
    n_word[len(dist_own)], n_word[len(d_good)], r0(d_good[0]), r0(d_good[-1]), d_bad_txt, n_word[len(d_out_bad)], sec(d_out_bad)))
html += ('<p>What it cannot do. <b>A hard bend cannot be undone.</b> Where the curve pressed many levels into one, nothing in the sound says which was which, so tops that were pressed flat stay flat. '
 'Far left puts the level relations right as far as that goes, and scores about %d on average over the 24 cases of the blind test on which DISTORTION is shown. It does not fill the tops in. That would take a declipper, which makes up what the tops were from the sound around them: another engine, and not in this build. '
 'The way back is careful on purpose. The reading is off by a dB or two, and spreading values apart that were never squeezed is a worse fault than leaving some of the bend in, so far left takes three quarters of what was read, less 1 dB. '
 '<b>A soft saturation is mostly not shown</b> (%d of %d sounds): it changes too little of the wave\'s shape to be told from the sound itself. '
 'A square wave or a pulse wave reads as a distorted sound, because that is what a sine pushed hard through a curve looks like. '
 'An effect that is more than a curve, with a filter before or behind it or with plain sound mixed in, is read less well, and under a loud room or behind a low-pass at 1 kHz the squeeze reads too low. '
 'Of the %d clean test sounds %d are shown, all with flat tops of their own, and of %d sounds through other effects %d.</p>'
 '<p><b>EXCITER: tested, and taken out.</b> An exciter adds highs to a sound: partly the highs it had, made louder, partly new ones. Up to build 0.5.0 the app could show an EXCITER group. '
 'It came up on one of the eight test cases, and what it then put on the second sound was a made-up exciter that left that sound far worse than untouched. For this build I tried to give it an engine like the others, and it cannot be done from one file. '
 'A reader taught for exactly this found %d of %d excited sounds, the heavy-handed ones, and took %d of the same %d sounds before the exciter for excited. '
 'A sound with 8 dB more highs than it had is, to anything that looks at that sound alone, a bright sound. So the group is gone. With two sounds, TONE MATCH carries the first sound\'s highs over to the second, which is what an exciter leaves in a spectrum.</p>'
 % (S['dist_out'], S['soft'][0], S['soft'][1], S['clean'], S['bend_clean'], S['bend_others'][1], S['bend_others'][0], S['exc_found'][0], S['exc_found'][1], S['exc_before'][0], S['exc_before'][1]))

# ---------------------------------------------------------------- the rest of the scoreboard
assert stray == [('dly_pedalboard_00', 'saturation'), ('exc_sox_00', 'distortion')], stray        # the sentences below name exactly these two
assert [m[0] for m in moved] == ['dly_pedalboard_00', 'pha_caps_00', 'trm_gx_01'], moved          # of the other 13 types' 104 cases these three changed
e = by['exc_sox_00']; d = by['dly_pedalboard_00']; d0 = byo['dly_pedalboard_00']; assert e['out_all'] < 0 and e['onto_all'] < 0 and d['out_all'] < d0['out_all'] and r0(d['onto_all']) == r0(d0['onto_all']) and d['found'] and d0['found']
gone = [(byo[n], by[n]) for n in ('pha_caps_00', 'trm_gx_01')]; assert all('exciter' in a['others'] and not b['others'] and a['onto_all'] < -100 and b['onto_all'] >= 0 and abs(a['onto'] - b['onto']) < 0.05 for a, b in gone)
html += ('<p><b>The rest of the scoreboard.</b> The new groups come up on two cases that are not theirs. One of the eight exciter cases is an exciter built from an overdrive on the highs, mixed under the sound. '
 'The app now shows DISTORTION there. It reads the mix as one curve, which it is not, and what it then does is worse than leaving the sounds alone (%s taken out, %s on a second sound). '
 'On one delay case it shows SATURATION as well, which costs that case a little when everything is taken out (%s, where it was %s) and nothing to speak of on a second sound (%s as before). '
 'With the EXCITER group gone, two cases that carried one that was not there, a phaser and a tremolo, no longer have their second sound made far worse by it. '
 'The other %d cases of the 13 remaining effect types come out exactly as in build 0.5.0.</p>'
 % (r0(e['out_all']), r0(e['onto_all']), r0(d['out_all']), r0(d0['out_all']), r0(d['onto_all']), 8 * len(rest) - len(moved)))
html += ('<p>The four examples below were picked by fixed rules, not by ear. For the filter: of the cases on which the group is shown, the middle one by its score for taking the filter out, and the one that goes onto a second sound worst. '
 'For DISTORTION and for SATURATION: of the cases shown under that name, the middle one by its score on a second sound.</p></section>')
if __name__ == '__main__': open(os.path.join(HERE, 'four051_head.html'), 'w').write(html); print('four051_head.html written,', len(html), 'characters')
