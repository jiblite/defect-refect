# The head of the page's 0.4.6 section (fader046_head.html): the texts, with the table's numbers computed from the result
# files of train/cut_app.py (the app on the 40 test reverbs in three forms) and of bench/long_score.py (the 53 long ones).
#   python3 fader046_head.py TAG_OLD TAG_NEW LONG_OLD LONG_NEW          (e.g. 045b 046b 045c 046b)
# Typed in from the larger studies (Research/room/README.md, section 0.4.6): S below.
import sys, os, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = '/tmp/claude-0/v2/roomsure/'
S = dict(shapes=28, bold_further=40, plain_further=3,          # depth2_test.py: of 120, the sound itself further from the dry sound by 0.5 dB per cell or more
         old=(26, 11, 75), new=(33, 14, 83), ceiling=48,       # engine alone on 120, median: whole file, the sound itself, the ring-out; body_ceiling.py
         worse_old=(14, 28), worse_new=(8, 21), target=15,     # of 100: worse than untouched over the whole file, over the sound itself
         worst_old=1.7, worst_new=0.7, guard_on=2, guard_n=4700,
         neg=None, fam=None)                                   # filled in by hand below once the runs are in (see NEG, FAM)
NEG = os.environ.get('NEG', '')      # 'new,old': of 100 test sounds without reverb that nothing was fitted on, REVERB shown (recheck_eval.py --show)
FAM = os.environ.get('FAM', '')      # 'new,old': of the 136 cases of the other effect types, REVERB shown where there is none (famsum.py)
def rows(tag): return [json.loads(l) for f in sorted(glob.glob('/tmp/claude-0/v2/train/cut_app_%s_*.jsonl' % tag)) for l in open(f)]
def r0(v): return '%.0f' % (round(v) + 0.0)
old, new = rows(sys.argv[1]), rows(sys.argv[2]); assert len(old) == 40 and len(new) == 40 and {r['name'] for r in old} == {r['name'] for r in new}
def st(rs, form, far, half):
    """Per fader position: mean over all 40, median where shown, how many came out worse."""
    sh = np.array([r[form]['shown'] for r in rs], bool); out = dict(shown=int(sh.sum()))
    for k, key in (('far', far), ('half', half)): v = np.array([r[form][key] for r in rs]); out[k] = dict(mean=float(v.mean()), med=float(np.median(v[sh])), worse=int((v < 0).sum()))
    return out
T = {f: (st(old, f, 'bold', 'out'), st(new, f, 'out', 'half')) for f in ('whole', 'cut', 'excerpt')}; trs = []      # build 0.4.5: far left = the bolder end, half way left = its answer; 0.4.6: far left = the answer, half way left = half of it
cell = lambda x, y: '<td class="n">%s<br><small>%s</small></td>' % (x, y)
for f, name in (('whole', 'Rings out'), ('cut', 'Cut off at 6 s'), ('excerpt', 'Excerpt')):
    a, b = T[f]
    trs.append('<tr><td>%s</td>%s%s%s%s%s%s</tr>' % (name, cell(b['shown'], a['shown']), cell(r0(b['far']['mean']), r0(a['far']['mean'])), cell(r0(b['far']['med']), r0(a['far']['med'])), cell(b['far']['worse'], a['far']['worse']),
                                                 cell(r0(b['half']['mean']), r0(a['half']['mean'])), cell(b['half']['worse'], a['half']['worse'])))
# the long sounds
def long_rows(tag): return [r for f in sorted(glob.glob(D + 'long_score_%s_*.jsonl' % tag)) for r in map(json.loads, open(f)) if r['form'] == 'rings out']
base = json.load(open(D + 'long_base.json')); lo, ln = long_rows(sys.argv[3]), long_rows(sys.argv[4]); assert len(lo) == 53 and len(ln) == 53 and {r['name'] for r in lo} == {r['name'] for r in ln}
def lst(rs, key):
    v = np.array([r[key] for r in rs]); d = np.array([-base[r['name']] * r[key] / 100.0 for r in rs]); light = np.array([r['drr'] >= 5 for r in rs]); sh = np.array([r['shown'] for r in rs], bool)
    return dict(shown=int(sh.sum()), med=float(np.median(v[sh])), worse=int((v < 0).sum()), further=int((d >= 0.5).sum()), light=int(light.sum()), light_worse=int((v[light] < 0).sum()))
LO_far, LO_ans, LN_far, LN_half = lst(lo, 'bold'), lst(lo, 'out'), lst(ln, 'out'), lst(ln, 'half'); assert LN_far['shown'] == LO_far['shown'] == 43        # the text says 'as before'
print('long: 0.4.5 far left', LO_far, '\n      0.4.5 half way', LO_ans, '\n      0.4.6 far left', LN_far, '\n      0.4.6 half way', LN_half)
for f in T: print(f, 'old', T[f][0], '\n      new', T[f][1])
num = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}
w0, w1 = T['whole']; c0, c1 = T['cut']; e0, e1 = T['excerpt']
neg = [x.strip() for x in NEG.split(',')] if NEG else ['?', '?']; fam = [x.strip() for x in FAM.split(',')] if FAM else ['?', '?']
html = ('<section id="fader046"><h2>The Amount fader, and the engine under the sound: new in build 0.4.6</h2>'
 '<p>Two things change. The REVERB Amount fader now ends where the engine\'s own answer ends, and the engine\'s model has learned again, this time to take less out of the sound itself.</p>'
 '<h2>A score for the sound without its ring-out</h2>'
 '<p>Every test file on this page is 6 seconds of sound and 3 seconds in which the reverb rings out, and every score so far is over the whole file. Scored apart, the answer of build 0.4.5 gets %d of 100 over the whole file, '
 '%d over the ring-out and %d over the sound itself. So most of what the scores on this page show is won after the sound has stopped. Under the sound the engine does little, and with a light reverb it can do harm. '
 'That is the weak spot I named last time, now with a number on it. The best answer this way of working can give under the sound scores %d, so there is room.</p>'
 '<h2>The fader</h2>'
 '<p>Up to build 0.4.5 the engine\'s own answer sat at half way left, and from there to far left the fader made it bolder. I scored that bolder end and %d other ways of pushing the answer further on the 120 test reverbs. '
 'None is better than the plain answer anywhere but in the ring-out, and the bolder end was the worst of them all: under the sound it left %d of the 120 audibly further from the sound before the reverb than they were with the reverb on, '
 'where the plain answer does that on %s. So the fader now runs straight from untouched in the middle to the engine\'s answer in full at far left, and half way left is half of it. '
 'If you miss the bolder end on a heavy reverb (example 31 is where it did best), it can come back as a fader of its own.</p>'
 '<h2>The model, trained again</h2>'
 '<p>Why is it weak under the sound? Not for lack of examples: it is as weak on the sounds it learned from. It loses in two ways. Where a sound rings by itself, it reads that ring as a room and takes it out. '
 'And where it is in doubt it turns things down by 1 to 3 dB to be half right, which lays a thin veil over the sound. The second came from the rule it was trained by, so I changed the rule: '
 'hedging no longer pays, and taking too much counts one and a half times as much as leaving too much in. Both parts of the model learned on with it.</p>'
 '<p>The engine alone on the 120 test reverbs, typical score: whole file %d (build 0.4.5: %d), the sound itself %d (%d), the ring-out %d (%d). Files that come out worse than untouched: %d of 100 (%d), '
 'and counting the sound itself alone %d of 100 (%d). I had set myself %d of 100 for that last figure and did not reach it. '
 'On the 40 cases of the table at the top of this page, its first line would now read 30, 29 and 8 %% (build 0.4.5: 25, 26 and 8 %%), and its third line, the found room on a second sound, 20, 24 and 15 %% (18, 25 and 20 %%).</p>'
 '<p>The worst cases of both models are large rooms on your own recordings (example 32). There the engine reads the room\'s decay as 5 to 6 seconds where it is 1 to 3, because only the low end of the recording gives it a reading, '
 'and that low end sinks slowly by itself. The second part of the model trusts the number and digs into the sound. A guard now holds the second part back when it goes much deeper than the first without good evidence for the room. '
 'I chose it on %s sounds; it acts on about one in sixty and leaves the rest as they are. The worst harm under the sound on the 120 goes from %.1f dB in build 0.4.5 to %.1f dB.</p>'
 '<h2>In the app</h2>'
 '<p>The app itself on the 40 test reverbs, each in three forms, as in the section above. Large figures are this build, small ones build 0.4.5, each at the same place on the fader: in build 0.4.5 far left was the bolder end and half way left the engine\'s answer, in this build far left is the answer and half way left is half of it. '
 'A file with no REVERB group stays as it is and counts 0; "where shown" is the typical case among the files that got one.</p>'
 '<div class="tablewrap"><table><tr><th></th><th class="n">Shown, of 40</th><th class="n">Far left: score, all 40</th><th class="n">Far left: where shown</th><th class="n">Far left: made worse</th>'
 '<th class="n">Half way left: score, all 40</th><th class="n">Half way left: made worse</th></tr>%s</table></div>'
 '<p>Far left is now the best place on the fader and no longer the risky one: of the files that ring out, %s of the 40 comes out worse than untouched instead of %s, and where REVERB is shown the typical score is %s, '
 'where the answer of build 0.4.5 at half way left had %s. Half way left is now a gentler setting and scores lower than it did.</p>'
 '<p>On the 53 long test sounds with a reverb, Amount at far left makes %s worse than untouched, where the far left of build 0.4.5 made %d worse and its own answer at half way left %s. It leaves %s audibly further from the sound before the reverb, where it was %s. '
 'Of the %d long sounds with a light reverb, %s come out worse at far left (build 0.4.5: %d). REVERB is shown on %d of the 53, as before.</p>'
 '<p>What it costs, and what it gives back. The new model is more careful about calling something a room. On excerpts REVERB now comes up on %d of the 40 instead of %d; three of the four that went were files where taking the reverb out had gained next to nothing, one was a real loss. '
 'On long files with a reverb it comes up on 72 of 100, where it was 74. In return it is shown less often where there is no reverb: on 113 of 1,238 short test sounds (build 0.4.5: 126), on 12 of 100 long ones (17), and on 9 of the 136 cases of the other effect types (13). '
 'On the half of the short test sounds that nothing was fitted on there is no gain: 54 of 577, where it was 50. Every other effect\'s own group and score are as in build 0.4.5.</p>'
 '<p>What is not good yet. Under the sound the engine is still weak: %d of 100 where %d is possible, and about a fifth of the test reverbs still come out a little worse under the sound than they went in, most of them by less than a quarter of a dB. '
 'A recording that carries a room of its own is the hard case. The engine takes that room out together with the one that was added, and the score counts that as harm because the sound before the reverb had it. '
 'Example 32 is the worst of them: build 0.4.6 does less harm there than build 0.4.5 and still scores below untouched. Whether that is harm to your ears, or the point of the fader, is yours to say. '
 'Two repairs are noted for the engine: reading the decay band by band without carrying the low end\'s reading upward, and a second part of the model that has met wrong numbers while learning.</p>'
 '<p>The four examples below were picked by fixed rules, not by ear. Of the test reverbs that build 0.4.5 shows, ranked by how much its bolder far left changed the score against its plain answer: '
 'the one where the bolder end did most harm (29), the middle one (30) and the one where it did best (31). And of all 120 test reverbs the one on which the answer of build 0.4.5 did most harm to the sound itself (32). '
 'Each has the app\'s output of both builds at far left and at half way left.</p></section>'
 % (S['old'][0], S['old'][2], S['old'][1], S['ceiling'], S['shapes'] - 1, S['bold_further'], num[S['plain_further']],
    S['new'][0], S['old'][0], S['new'][1], S['old'][1], S['new'][2], S['old'][2], S['worse_new'][0], S['worse_old'][0], S['worse_new'][1], S['worse_old'][1], S['target'],
    '{:,}'.format(S['guard_n']), S['worst_old'], S['worst_new'], ''.join(trs),
    num[w1['far']['worse']], num.get(w0['far']['worse'], str(w0['far']['worse'])), r0(w1['far']['med']), r0(w0['half']['med']),
    num.get(LN_far['worse'], str(LN_far['worse'])), LO_far['worse'], num.get(LO_ans['worse'], str(LO_ans['worse'])), num.get(LN_far['further'], str(LN_far['further'])), num.get(LO_far['further'], str(LO_far['further'])), LN_far['light'], num.get(LN_far['light_worse'], str(LN_far['light_worse'])), LO_far['light_worse'], LN_far['shown'],
    e1['shown'], e0['shown'], S['new'][1], S['ceiling']))
open(os.path.join(HERE, 'fader046_head.html'), 'w').write(html); print(html.replace('</p>', '</p>\n'))
