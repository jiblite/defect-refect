# The head of the page's 0.4.5 section (cut045_head.html): the texts, with the table's numbers computed from the result
# files of train/cut_app.py (the app on the 40 test reverbs in three forms) and of the scoreboard (bench/famboard.py).
#   python3 cut045_head.py TAG_OLD TAG_NEW FAMBOARD_OLD.jsonl FAMBOARD_NEW.jsonl
# Typed in from the larger studies (Research/room/README.md, section 0.4.5): S below.
import sys, os, json, glob, numpy as np
sys.path.insert(0, '/tmp/claude-0/v2/roomsure'); sys.path.insert(0, '/tmp/claude-0/v2/bench'); import look2, famsum
HERE = os.path.dirname(os.path.abspath(__file__))
S = dict(engine_cut_worse=48,                     # cut_eval.py, the 0.4.4 pair on 40 reverbs cut off at 6 s: share made worse than untouched
         renders=3615, looks=2470,                # mkcases_x.py; fit_sure.py
         neg_n=577, neg_old=14.0, neg_new=8.7,    # sounds without reverb, unseen half, REVERB shown at 65 %: look2.py 044; recheck_eval.py --show b
         fx_n=912, fx_old=(134, 86, 48), fx_new=(93, 67, 26),      # effect alone: shown, of them on a clip that shows it clean, made by the effect
         pos_shown=340, pos_again=78, pos_lost=1, # recheck_eval.py: sounds with a real reverb
         long_n=134, long_neg=(121, 40, 17), long_pos=(144, 76, 74),     # long_eval3.py: long test sounds; looks without reverb: how many, shown of 100 by the look at the whole sound and now; the same with reverb
         long_clean=(12, 9, 10, 3),               # the twelve clean rows of clips: shown by build 0.4.4 (the app itself), by the look at the whole sound, and now
         long_out=(53, 43, 32, 6, 17, 16, 1, 5))  # long_score.py: long sounds with a reverb, shown; of them light (direct 5 dB and more over the reverb), worse than untouched half way left and far left; the 9 s test sounds: light, worse half way left, far left
def rows(tag): return [json.loads(l) for f in sorted(glob.glob('/tmp/claude-0/v2/train/cut_app_%s_*.jsonl' % tag)) for l in open(f)]
def r0(v): return '%.0f' % (round(v) + 0.0)
old, new = rows(sys.argv[1]), rows(sys.argv[2]); assert len(old) == 40 and len(new) == 40 and {r['name'] for r in old} == {r['name'] for r in new}
def st(rs, form):
    o = np.array([r[form]['out'] for r in rs]); b = np.array([r[form]['bold'] for r in rs]); sh = np.array([r[form]['shown'] for r in rs], bool)
    return dict(shown=int(sh.sum()), mean=float(o.mean()), med=float(np.median(o[sh])), worse=int((o < 0).sum()), worse_left=int((b < 0).sum()))
T = {f: (st(old, f), st(new, f)) for f in ('whole', 'cut', 'excerpt')}; trs = []
for f, name in (('whole', 'Rings out'), ('cut', 'Cut off at 6 s'), ('excerpt', 'Excerpt')):
    a, b = T[f]; cell = lambda x, y: '<td class="n">%s<br><small>%s</small></td>' % (x, y)
    trs.append('<tr><td>%s</td>%s%s%s%s%s</tr>' % (name, cell(b['shown'], a['shown']), cell(r0(b['mean']), r0(a['mean'])), cell(r0(b['med']), r0(a['med'])), cell(b['worse'], a['worse']), cell(b['worse_left'], a['worse_left'])))
# the scoreboard: REVERB shown on a case whose effect is another one; how many of those sit on a clip that shows REVERB with no effect on it
fo, fn = famsum.load(sys.argv[3]), famsum.load(sys.argv[4]); so, sn = famsum.summ(fo), famsum.summ(fn); assert so['n'] == 136 and sn['n'] == 136
sb = look2.load('sb', 'x45'); clean = {r['case']: r['v_as'] for r in sb if r['fam'] == 'clean'}; dry = {r['case']: r['dry'] for r in sb if r['fam'] not in ('clean', 'reverb', 'tone') and r['which'] == 'A'}
false_new = [r['name'] for r in fn if r['fam'] != 'reverb' and 'reverb' in r['others']]; own = [n for n in false_new if clean[dry[n]] > 0.65]
print('scoreboard false REVERB %d -> %d, of them on a clip that shows REVERB clean: %d; made by the effect: %s' % (so['reverb'], sn['reverb'], len(own), sorted(set(false_new) - set(own))))
num = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}
c0, c1 = T['cut']; e0, e1 = T['excerpt']; w0, w1 = T['whole']
html = ('<section id="cut045"><h2>Files that stop while the sound plays: new in build 0.4.5</h2>'
 '<p>The reverb engine\'s model had only ever learned from sounds that ring out inside the file. A file that stops while the sound still plays, or an excerpt cut at both ends, gives it no tail to read, and there it did as much harm as good: '
 'on 40 test reverbs cut off at 6 seconds its own answer made %d of 100 worse than leaving them alone. Build 0.4.4 kept most of that off the screen, because its look with silence after the sound hid the REVERB group on about half of such files.</p>'
 '<p>The model has now learned again, from %s new examples. Each one is there in three forms: ringing out, cut off at 6 seconds, and as a 4.5 second excerpt. Over half of them have another effect on the sound before the reverb, '
 'and 45 of 100 have no reverb at all, so that it also learns where there is nothing to find. The Sureness figure on the screen was fitted again on %s test sounds, and the look with silence is gone, because the new model does not need it.</p>'
 '<p>The app itself on the 40 test reverbs, each in three forms: ringing out inside the file, cut off at 6 seconds, and as an excerpt cut at both ends. "Shown" is on how many of the 40 the REVERB group comes up. The scores are for REVERB Amount half way left (100 = the sound before the reverb, 0 = no better than untouched), and a file with no REVERB group stays as it is and counts 0. '
 '"All 40" is the average with those zeros in it, "where shown" is the typical case among the files that got a REVERB group. The last two columns count the files that came out worse than they went in. Large figures are this build, small ones build 0.4.4.</p>'
 '<div class="tablewrap"><table><tr><th></th><th class="n">Shown, of 40</th><th class="n">Score, all 40</th><th class="n">Score where shown</th><th class="n">Made worse</th><th class="n">Worse at far left</th></tr>%s</table></div>'
 '<p>On a cut-off file the REVERB group now comes up on %d of the 40 instead of %d, and where it comes up it does more: %s against %s. On excerpts it is %d instead of %d, and %s against %s. Sounds that ring out gain a little: the first line of the table at the top of this page would now read 25, 26 and 8 %% instead of 23, 23 and 12 %%.</p>'
 '<h2>A reverb that is not there</h2>'
 '<p>I looked at where the false REVERB of build 0.4.4 came from. On %d test sounds with one other effect and no reverb it showed REVERB on %d. On %d of those the ring is in the sound before any effect: recorded loops and sampled instruments '
 '(plucked strings, steel drum, vibraphone, choir) that ring by themselves or carry the room they were recorded in. The app cannot know that nobody added that ring later. The other %d were made by the effect: an echo, a delay between left and right, a chorus or a widening can look like a room.</p>'
 '<p>For that second kind the app now looks twice. When it finds a room together with an effect it knows how to take out (echoes, tremolo, auto pan, gate, flanger, phaser, filter sweep, chorus, a delay between left and right), '
 'it takes that effect out and looks for the room again, and the less sure of the two answers counts. Together with the new model that brings the false REVERB made by an effect from %d to %d, and the kind that is in the sound itself from %d to %d. '
 'Of %d sounds with a real reverb, %d were looked at again and %s lost its REVERB group.</p>'
 '<p>All told, on %d test sounds without reverb that neither the model nor the Sureness figure was fitted on, REVERB is shown on %s of 100, where build 0.4.4 showed it on %s. '
 'On the 136 scoreboard cases it is shown where there is none on %d instead of %d, and %s of those %d are sounds that show it with no effect on them.</p>'
 '<h2>Long files</h2>'
 '<p>Everything above is measured on sounds of 6 and 9 seconds, which is also what the reverb model learned from. Most real files are longer, so I made %d long test sounds of 18 to 48 seconds, with and without reverb, '
 'and found a fault that every build so far has had: when the app looks at a whole long sound, it reads a room into it far too easily. Before the repair this build showed REVERB on %d of 100 long sounds without any reverb. '
 '%s of them are plain clean clips in a row, and on those build 0.4.4 showed it on %s and this build at first on %s.</p>'
 '<p>A sound of more than 9 seconds is now judged in pieces of 6 seconds, each looked at as a sound of its own. It counts as having a room when the pieces say so on average, or when the last piece does, because the end is where a sound rings out. '
 'That brings the false REVERB on the long sounds from %d to %d of 100, and on the %s clean ones to %s, each of which ends on a sound that rings by itself. Of 100 long sounds with a real reverb %d are shown, where it was %d. '
 'Sounds of up to 9 seconds are judged exactly as before, so nothing else on this page changes.</p>'
 '<p>I also scored for the first time how well the reverb comes out of a long file, on %d long sounds with a reverb, of which %d get a REVERB group. A heavy reverb comes out about as well as on the short test sounds. '
 'A light reverb, where the sound itself is 5 dB or more above its reverb, does not. Of %d long sounds with a light reverb, %s come out worse than untouched with Amount half way left, and %d with Amount at far left; on the short test sounds that was %s of %d and %s of %d. '
 'Length is not the main reason. With a light reverb the engine can take more out of the sound itself than the reverb put in, and it gains most where the sound rings out at the end. '
 'A short test sound is one third ring-out and a long file is nearly all sound, so the scores of short sounds on this page flatter the engine there. '
 'Working through a long file in pieces was tried and does not repair it. Until the engine is better there, half way left is the safer setting for a light reverb, and far left is not.</p>'
 '<p>What is not good yet. On cut-off files and excerpts the engine still does not see the room on about a third of the real reverbs, light ones most; those files stay as they are. '
 'Amount at far left is still a risk on every kind of file and makes about a fifth to a quarter of the cases worse than untouched, where half way left makes one or two of 40 worse. '
 'Sounds that ring by themselves still read as rooms. And widening, distortion and ring modulation, which have no engine of their own yet, can still make a REVERB that is not there. A larger model is the next lever for the first two. '
 'And light reverbs, as described under long files, are now the weakest spot of the reverb engine.</p>'
 '<p>The five examples below were picked by a fixed rule from the 40 test reverbs, not by ear: of the cut-off files that both builds show, the third largest change and the middle one (24, 25), '
 'the middle one of those that only build 0.4.5 shows (26), the one that only build 0.4.4 showed (27), and of the excerpts that both show, the middle one (28). Each has the app\'s output of both builds.</p></section>'
 % (S['engine_cut_worse'], '{:,}'.format(S['renders']), '{:,}'.format(S['looks']), ''.join(trs),
    c1['shown'], c0['shown'], r0(c1['med']), r0(c0['med']), e1['shown'], e0['shown'], r0(e1['med']), r0(e0['med']),
    S['fx_n'], S['fx_old'][0], S['fx_old'][1], S['fx_old'][2], S['fx_old'][2], S['fx_new'][2], S['fx_old'][1], S['fx_new'][1], S['pos_shown'], S['pos_again'], num[S['pos_lost']],
    S['neg_n'], r0(S['neg_new']), r0(S['neg_old']), sn['reverb'], so['reverb'], num.get(len(own), str(len(own))), sn['reverb'],
    S['long_n'], S['long_neg'][1], num[S['long_clean'][0]].capitalize(), S['long_clean'][1], S['long_clean'][2],
    S['long_neg'][1], S['long_neg'][2], num[S['long_clean'][0]], num[S['long_clean'][3]], S['long_pos'][2], S['long_pos'][1],
    S['long_out'][0], S['long_out'][1], S['long_out'][2], num[S['long_out'][3]], S['long_out'][4], num[S['long_out'][6]], S['long_out'][5], num[S['long_out'][7]], S['long_out'][5]))
open(os.path.join(HERE, 'cut045_head.html'), 'w').write(html); print(html.replace('</p>', '</p>\n'))
