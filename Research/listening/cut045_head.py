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
         pos_shown=340, pos_again=78, pos_lost=1) # recheck_eval.py: sounds with a real reverb
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
 '<p>What is not good yet. On cut-off files and excerpts the engine still does not see the room on about a third of the real reverbs, light ones most; those files stay as they are. '
 'Amount at far left is still a risk on every kind of file and makes about a fifth to a quarter of the cases worse than untouched, where half way left makes one or two of 40 worse. '
 'Sounds that ring by themselves still read as rooms. And widening, distortion and ring modulation, which have no engine of their own yet, can still make a REVERB that is not there. A larger model is the next lever for the first two.</p>'
 '<p>The five examples below were picked by a fixed rule from the 40 test reverbs, not by ear: of the cut-off files that both builds show, the third largest change and the middle one (24, 25), '
 'the middle one of those that only build 0.4.5 shows (26), the one that only build 0.4.4 showed (27), and of the excerpts that both show, the middle one (28). Each has the app\'s output of both builds.</p></section>'
 % (S['engine_cut_worse'], '{:,}'.format(S['renders']), '{:,}'.format(S['looks']), ''.join(trs),
    c1['shown'], c0['shown'], r0(c1['med']), r0(c0['med']), e1['shown'], e0['shown'], r0(e1['med']), r0(e0['med']),
    S['fx_n'], S['fx_old'][0], S['fx_old'][1], S['fx_old'][2], S['fx_old'][2], S['fx_new'][2], S['fx_old'][1], S['fx_new'][1], S['pos_shown'], S['pos_again'], num[S['pos_lost']],
    S['neg_n'], r0(S['neg_new']), r0(S['neg_old']), sn['reverb'], so['reverb'], num.get(len(own), str(len(own))), sn['reverb']))
open(os.path.join(HERE, 'cut045_head.html'), 'w').write(html); print(html.replace('</p>', '</p>\n'))
