# Writes the page's three text blocks (intro.html, table.html, outro.html) with the numbers of the scored runs filled in.
#   MODEL=<name> python3 texts.py
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); MODEL = os.environ.get('MODEL', 'unet_c'); T = '/tmp/claude-0/v2/train/'
rows = json.load(open(T + 'final_eval_%s.json' % MODEL)); best = json.load(open(T + 'final_eval_oracle.json')); RANKS = os.environ.get('RANKS', '3,10,17,24,31,38').split(',')
# the same cases split by the sound under the reverb: mono or narrow against wide stereo (source_width.json: how much of the dry
# sound left and right do not share, dB under the whole; under -20 counts as mono or narrow)
OWN = json.load(open(os.path.join(HERE, 'source_width.json'))); mono = np.array([r['out']['reverb out'] for r in rows if OWN[r['name']] < -20]); wide = np.array([r['out']['reverb out'] for r in rows if OWN[r['name']] >= -20])
def line(rs, key):
    v = np.array([r['out'][key] for r in rs]); return '<td class="n">%.0f</td><td class="n">%.0f</td><td class="n">%.0f %%</td>' % (np.median(v), np.mean(v), 100 * np.mean(v < 0))
tail = np.nanmedian([r['tail_cut'] for r in rows]); body = np.nanmedian([r['body_cut'] for r in rows]); n = len(rows); tail2 = np.nanmedian([r['tail_cut_bold'] for r in rows]); body2 = np.nanmedian([r['body_cut_bold'] for r in rows])
open(os.path.join(HERE, 'intro.html'), 'w').write('''<p class="lead">This is the V2 reverb engine as it runs in build 0.4.1, with the tone engine further down. It hears one sound and decides, band by band and moment by moment, what is room and what is the sound itself. From that it reads the room (decay time, level, width) and can turn the room down or up on that sound, or put the same room on a second sound.</p>
<p class="lead">Every reverb here comes from an effect that is not my code, and the engine never gets the dry sound. Headphones help.</p>''')
open(os.path.join(HERE, 'table.html'), 'w').write('''<section class="panel"><h2>By the numbers</h2>
<p>%d test cases, each scored against the true answer. 100 = identical to it, 0 = no better than leaving the sound alone, below 0 = worse than doing nothing.</p>
<div class="tablewrap"><table>
<tr><th></th><th class="n">Typical case</th><th class="n">Average</th><th class="n">Made worse</th></tr>
<tr><td>Reverb taken out of a sound (Amount half way left)</td>%s</tr>
<tr><td>The same with a perfect room-or-sound decision</td>%s</tr>
<tr><td>Found room put on a second sound</td>%s</tr>
<tr><td>The same with a perfect decision</td>%s</tr>
</table></div>
<p>In plain terms: with Amount half way left, the room comes down by about %.0f dB where it is on its own (tails and gaps), and the sound loses about %.1f dB where it is on top. That is the engine's own answer. Further left it digs deeper: at far left the room comes down by about %.0f dB and the sound loses about %.0f dB. How deep is right depends on the sound, so the fader leaves it to your ear. The old app could not take reverb out at all, and my best hand-made rule scored 12.</p>
<p>The second pair of lines shows where the limits are. Taking reverb out is held back by the decision, and that is where the next gains are. The room on a second sound is a copy by numbers (decay time, level, tone, width), not a recording of the real room, so it cannot reach 100: with the true numbers it scores about 32.</p>
<p>The weak spot of the first model (build 0.4.0): it told room from sound largely by what left and right do not share. The model in this build was also trained on sounds where that cue is missing. With a mono or narrow sound under the reverb it does well (typical score %.0f, %.0f %% of cases made worse). With a wide stereo sound under it, it still takes parts of the sound for room more often (average %.0f, %.0f %% made worse). The clear gain is on recordings that are narrow or mono as a whole, room included: there it now finds the room on about 19 in 20 sounds (before: 8 in 10) and takes out about twice as much as before.</p>
<p>The six examples are places %s of the %d cases ranked by the first line: the whole range from good to bad, not my favourites.</p></section>''' % (
    n, line(rows, 'reverb out'), line(best, 'reverb out'), line(rows, 'room onto second sound'), line(best, 'room onto second sound'), tail, body, tail2, body2, np.median(mono), 100 * np.mean(mono < 0), np.mean(wide), 100 * np.mean(wide < 0), ', '.join(RANKS[:-1]) + ' and ' + RANKS[-1], n))
open(os.path.join(HERE, 'outro.html'), 'w').write('''<h2>What comes next</h2>
<p>Build 0.4.1 has all of this: the REVERB group with the new model, the Tone Match button, the COMB FILTER group, and an On/Off switch on every effect. Next: fewer effects shown that are not there (a new detector is in work), and a new echo engine that measures the echo itself.</p>
<p>What helps most from your ears: which Amount position sounds right for taking the reverb out (half way left, far left, or somewhere between), and does the room on the second sound feel like the same room?</p>''')
print('texts written for', MODEL, '| tail %.1f dB, body %.2f dB' % (tail, body))
