# The two blocks of numbers in the head of the page's 0.4.3 section, computed from the result files:
#   python3 level_numbers.py FAMBOARD_042.jsonl FAMBOARD_043.jsonl LEVEL_TAG [SECOND_DRAW_TAG]
# reads level_head_tpl.html (placeholders NUMBERS-SHORT, NUMBERS-LONG), writes level_head.html and puts the new head into level.html.
import sys, os, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/v2/level'); sys.path.insert(0, '/tmp/claude-0/v2/sweep'); import lsum
HERE = os.path.dirname(os.path.abspath(__file__))
def fam(path): 
    d = {}
    for r in map(json.loads, open(path)): d.setdefault(r['fam'], []).append(r)
    return d
def r0(v): return '%.0f' % (round(v) + 0.0)          # no "-0"
old, new = fam(sys.argv[1]), fam(sys.argv[2]); tag = sys.argv[3]; tag2 = sys.argv[4] if len(sys.argv) > 4 else None
rows = []
for key, name in (('tremolo', 'Tremolo'), ('autopan', 'Auto pan'), ('gatedrhythm', 'Gated rhythm'), ('flanger', 'Flanger')):
    a, b = old[key], new[key]; med = lambda rs, k: float(np.median([r[k] for r in rs]))
    rows.append('<tr><td>%s</td><td class="n">%d of %d<br><small>%d</small></td><td class="n">%s<br><small>%s</small></td><td class="n">%s<br><small>%s</small></td></tr>'
                % (name, sum(r['found'] for r in b), len(b), sum(r['found'] for r in a), r0(med(b, 'out')), r0(med(a, 'out')), r0(med(b, 'onto')), r0(med(a, 'onto'))))
fl = sorted([r for r in new['flanger'] if r['found']], key=lambda r: -r['out'])
short = ('<p>The test cases of 6 seconds: eight per effect, made with effects that are not my code, each scored against the true answer (100 = identical to it, 0 = no better than leaving the sound alone). '
         'Large figures are this build, small ones build 0.4.2; a case that is not shown counts 0.</p>'
         '<div class="tablewrap"><table><tr><th></th><th class="n">Shown</th><th class="n">Taken out</th><th class="n">Onto a second sound</th></tr>%s</table></div>'
         '<p>The four flangers that are found come out at %s. On 198 clean test sounds none of these four groups is shown.</p>'
         % (''.join(rows), ', '.join(r0(r['out']) for r in fl[:-1]) + ' and ' + r0(fl[-1]['out'])))
S = lsum.summary(tag); w, p, c, sl = S['wet'], S['part'], S['clean'], S['slow']
def three(d, k): return 'tremolo %s, auto pan %s, gate %s' % tuple(r0(d[f][k]) for f in ('tremolo', 'autopan', 'gatedrhythm'))
long_ = ('<p>Long sounds, 20 to 62 seconds. Level movement: on %d sounds with the effect on all of the sound it is shown on %d and taken out at %s (typical cases). '
         'On %d sounds with the effect on one half it is shown on %d and taken out at %s typically, none comes out worse than it went in, and the half without the effect comes back exactly as it went in on %d of those %d. '
         'Slow pans, 3 to 23 seconds a sweep, are shown on %d of %d and taken out at %s%s. On %d clean long sounds nothing is shown.'
         % (w['all']['n'], w['all']['shown'], three(w, 'out_med'), p['all']['n'], p['all']['shown'], r0(p['all']['out_med']), p['off']['exact'], p['off']['n'], sl['all']['shown'], sl['all']['n'], r0(sl['all']['out_shown']),
            '; the other %s pass only about once in their sound and are offered when Sureness is turned down' % {1: 'one', 2: 'two', 3: 'three', 4: 'four'}.get(sl['all']['lower'], sl['all']['lower']) if sl['all']['lower'] == sl['all']['n'] - sl['all']['shown'] and sl['all']['lower'] else '', c['n']))
assert c['shown'] == 0 and p['all']['out_worse'] == 0
if tag2:
    S2 = lsum.summary(tag2); w2, p2, c2 = S2['wet'], S2['part'], S2['clean']
    assert c2['shown'] == 0 and w2['all']['out_worse'] == 0 and p2['all']['out_worse'] == 0
    long_ += ('</p><p>A second draw of long sounds gives the same picture: shown on %d of %d and %d of %d, taken out at %s, none worse than it went in, nothing shown on %d clean ones.'
              % (w2['all']['shown'], w2['all']['n'], p2['all']['shown'], p2['all']['n'], three(w2, 'out_med'), c2['n']))
    third = '/tmp/claude-0/v2/level/fast_third_k.jsonl'
    if os.path.exists(third):
        R = [json.loads(l) for l in open(third)]; ok = lambda v: v.get('found') and v.get('score') is not None and v['score'] >= 0 and v['kind'] == 'together'
        def med3(mode, t): return float(np.median([r['v'][t]['out'] if ok(r['v'][t]) else 0.0 for r in R if r['mode'] == mode and r['fam'] == 'gatedrhythm']))
        worse = sum(ok(r['v']['k']) and r['v']['k']['out'] < 0 for r in R if r['fam'] == 'gatedrhythm')
        assert worse == 0 and med3('wet', 'k') > med3('wet', 'i') and med3('part', 'k') > med3('part', 'i')
        long_ += (' That second draw also brought out three faults with gates, which were fixed before this build. On a third draw, made after the fixes, gates come out at %s on a whole sound and at %s on one half; '
                  'the code before the fixes had %s and %s there.' % (r0(med3('wet', 'k')), r0(med3('part', 'k')), r0(med3('wet', 'i')), r0(med3('part', 'i'))))
        anyk = lambda v: v.get('found') and v.get('score') is not None and v['score'] >= 0
        trm = [r for r in R if r['fam'] == 'tremolo' and r['mode'] in ('wet', 'part') and anyk(r['v']['k']) and r['v']['k']['out'] < 0]
        pan = [r for r in R if r['fam'] == 'gatedrhythm' and anyk(r['v']['k']) and r['v']['k']['kind'] != 'together']
        assert len(trm) == 1 and len(pan) == 1
        long_ += ' Two things went wrong on that third draw: one tremolo comes out worse than it went in, and one gated sound is shown as a slow auto pan instead of a gate.'
long_ += '</p>'
def frows(name): return sorted(map(json.loads, open('/tmp/claude-0/v2/sweep/flong_%s_f.jsonl' % name)), key=lambda r: r['i'])
def fshown(r): return r['found'] and (r['sounding'] < 2 or r['shown'] >= 2)
fw, fp, fc = frows('wet'), frows('part'), frows('clean'); lst = lambda rs: ', '.join(r0(r['out']) for r in rs[:-1]) + ' and ' + r0(rs[-1]['out'])
long_ += ('<p>Flanger on long sounds: found less often. On %d flanged sounds it is shown on %d (taken out at %s) and on %d more when Sureness is turned down, and none comes out worse than it went in. '
          'On %d sounds flanged in one half it is shown on %d (taken out at %s%s), and the half without the flanger comes back as it went in. On %d clean long sounds no flanger is shown.</p>'
          % (len(fw), sum(map(fshown, fw)), lst(sorted(filter(fshown, fw), key=lambda r: -r['out'])), sum(r['found'] and not fshown(r) for r in fw),
             len(fp), sum(map(fshown, fp)), lst(sorted([r for r in fp if fshown(r) and r['out'] >= 0], key=lambda r: -r['out'])),
             '; %s comes out a little worse than it went in' % {1: 'one', 2: 'two'}[sum(fshown(r) and r['out'] < 0 for r in fp)] if any(fshown(r) and r['out'] < 0 for r in fp) else '', len(fc)))
assert not any(fshown(r) for r in fc) and not any(fshown(r) and r['out'] < 0 for r in fw)
tpl = open(os.path.join(HERE, 'level_head_tpl.html')).read(); assert tpl.count('<p>NUMBERS-SHORT</p>') == 1 and tpl.count('<p>NUMBERS-LONG</p>') == 1
head = tpl.replace('<p>NUMBERS-SHORT</p>', short).replace('<p>NUMBERS-LONG</p>', long_); open(os.path.join(HERE, 'level_head.html'), 'w').write(head)
lv = os.path.join(HERE, 'level.html'); t = open(lv).read(); k = t.index('</section>') + len('</section>'); assert t.startswith('<section id="level">')
open(lv, 'w').write(head.rstrip('\n') + t[k:]); print(short); print(long_)
