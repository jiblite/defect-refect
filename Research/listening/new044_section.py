# The page's section for build 0.4.4 (chorus, phaser, filter sweep): six examples rendered by the app's own processing (the
# command-line tool drrender), as MP3s in audio/nw_*.mp3, and the section's HTML in new044.html (build.py puts it after the
# level section).
#   python3 new044_section.py
import os, sys, re, json, html, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); import score, truth
HERE = os.path.dirname(os.path.abspath(__file__)); AUD = os.path.join(HERE, 'audio'); SR = 48000; TMP = '/dev/shm/np/'; os.makedirs(TMP, exist_ok=True)
EXE = '/tmp/claude-0/drbuild/drrender'; RES = '/home/claude/defect-refect/Resources'; SETS = '/tmp/claude-0/v2/bench/sets/'
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
E = html.escape
def mp3(name, x, gain):
    wav = os.path.join(AUD, name + '.wav'); y = np.clip(x * gain, -1.0, 1.0); n = len(y); f = min(480, n // 4); y[:f] *= np.linspace(0, 1, f)[:, None]; y[-f:] *= np.linspace(1, 0, f)[:, None]
    sf.write(wav, y, SR, subtype='PCM_24'); subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame', '-b:a', '256k', os.path.join(AUD, name + '.mp3')], check=True); os.remove(wav)
    return 'audio/%s.mp3' % name
def row(cls, label, small, src): return '<div class="row %s"><div class="lab">%s<small>%s</small></div><audio controls preload="none" src="%s"></audio></div>' % (cls, E(label), E(small), src)
def run(first, blocks):
    args = []
    for i, (second, out, fad) in enumerate(blocks): args += (['--'] if i else []) + [second, out] + list(fad)
    r = subprocess.run([EXE, RES, first] + args, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
    return r.stdout
def rd(p, n): y, _ = sf.read(p, dtype='float64', always_2d=True); return y[:n]
def render(group, tag, wetA, dryA, dryB, wetB, extras, cut=None, cutb=None):
    """The app on the first sound (wetA) alone and with the second (dryB). cut = (from, to) in s: only that stretch goes to the page."""
    nA, nB = len(wetA), len(dryB); pa, pb = TMP + tag + '_a.wav', TMP + tag + '_b.wav'; sf.write(pa, wetA, SR, subtype='FLOAT'); sf.write(pb, dryB, SR, subtype='FLOAT')
    others = [k + '.amount=0' for k in ALL if k != group]
    blocks = [('-', TMP + tag + '_o0.wav', [group + '.amount=0']), ('-', TMP + tag + '_o1.wav', [k + '.amount=0' for k in ALL]), (pb, TMP + tag + '_o2.wav', others), (pb, TMP + tag + '_o3.wav', [])]
    for i, (_, _, fad) in enumerate(extras): blocks.append((pb, TMP + tag + '_x%d.wav' % i, others + fad))
    log = run(pa, blocks); shown = re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', log, re.M)
    A = dict(wet=wetA, out=rd(blocks[0][1], nA), out_all=rd(blocks[1][1], nA), dry=dryA); B = dict(b_dry=dryB, b_onto=rd(blocks[2][1], nB), b_all=rd(blocks[3][1], nB))
    if wetB is not None: B['b_real'] = wetB
    for i in range(len(extras)): B['x%d' % i] = rd(TMP + tag + '_x%d.wav' % i, nB)
    sc = dict(out=score.removed(dryA, wetA, A['out'])[0], out_all=score.removed(dryA, wetA, A['out_all'])[0])
    if wetB is not None: sc.update(onto=score.removed(wetB, dryB, B['b_onto'])[0], onto_all=score.removed(wetB, dryB, B['b_all'])[0])
    full = dict(A); full.update(B)
    if cut is not None: a, b = int(cut[0] * SR), int(cut[1] * SR); A = {k: v[a:b] for k, v in A.items()}; a, b = (int(v * SR) for v in (cutb or cut)); B = {k: v[a:b] for k, v in B.items()}
    x = dict(A); x.update(B); g = 0.89 / max(np.abs(v).max() for v in x.values()); f = {k: mp3('nw_%s_%s' % (tag, k), v, g) for k, v in x.items()}
    cho = re.search(r'^chorus: (.*)$', log, re.M); mov = re.search(r'^moving filter: (.*)$', log, re.M)
    print(tag, 'shown:', shown, '|', cho.group(0)[:200] if cho else None, '|', mov.group(0) if mov else None, '| scores', {k: round(v, 1) for k, v in sc.items()}, flush=True)
    return f, sc, [s for s, _ in shown], dict(chorus=cho.group(1) if cho else '', moving=mov.group(1) if mov else '', sure=dict(shown)), full
def panel(title, meta, one, rows1, two, rows2, more):
    return ('<section class="panel"><h3>%s</h3><div class="meta">%s</div><div class="sub">%s</div><div class="rows">%s</div><div class="sub">%s</div><div class="rows">%s</div>%s</section>'
            % (E(title), meta, E(one), ''.join(rows1), E(two), ''.join(rows2), ('<details><summary>The other faders on this example</summary><div class="rows">%s</div></details>' % more) if more else ''))
NAME = dict(chorus='chorus', phaser='phaser', filtersweep='filter sweep'); GROUP = dict(chorus='CHORUS', phaser='PHASER', filtersweep='FILTER SWEEP')
parts = []; info = {}
def spans_of(s): m = re.search(r'there_s=([0-9.,e+-]+)', s); return [tuple(float(v) for v in p.split('..')) for p in (m.group(1).split(',') if m else []) if '..' in p]
def clockt(s): s = int(round(s)); return '%d:%02d' % (s // 60, s % 60)
def found_text(group, d):
    if group == 'chorus':
        v = re.findall(r'\[([LR]) (sin|tri) ([0-9.e+-]+) Hz ([0-9.e+-]+)\+-([0-9.e+-]+) ms loud ([0-9.e+-]+)( ahead)?\]', d['chorus']); L = [q for q in v if q[0] == 'L'] or v
        lo = min(float(q[3]) - float(q[4]) for q in L); hi = max(float(q[3]) + float(q[4]) for q in L); loud = float(np.mean([(1.0 / max(float(q[5]), 0.25)) if q[6] else float(q[5]) for q in L]))     # a copy louder than the sound: the number holds the sound's level against the copy
        lev = ('about %.1f times the sound\'s level' % loud) if loud > 1.5 else ('about %.0f %% of the sound\'s level' % (100 * loud))
        return 'The app found %s a side, %s, %s between %.1f and %.1f ms, %.2f times a second.' % ('one copy' if len(L) == 1 else '%d copies' % len(L), ('at ' + lev) if len(L) == 1 else ('each at ' + lev),
                'its delay moving' if len(L) == 1 else 'their delays moving', lo, hi, float(L[0][2]))
    m = re.search(r'sure=([0-9.e+-]+) period_s=([0-9.e+-]+)', d['moving']); per = float(m.group(2)) if m else 0.0
    return 'The app found a %s%s.' % (NAME[group], (' whose curve comes round every %.1f seconds' % per) if per > 0 else '')
def short(group, fam_case, first, tag, title, real, second, extras, sounds=None):
    if sounds is None:
        fam, case = fam_case; cA = truth.load(SETS + fam + '/' + case + '/', first); cB = truth.load(SETS + fam + '/' + case + '/', 'B' if first == 'A' else 'A'); wetA, dryA, dryB, wetB = cA['wet'], cA['dry'], cB['dry'], cB['wet']
    else: wetA, dryA, dryB, wetB = sounds
    f, sc, shown, d, _ = render(group, tag, wetA, dryA, dryB, wetB, extras); info[tag] = dict(scores=sc, shown=shown, found=d)
    others = [s for s in shown if s != group]; nm = NAME[group]
    note = '' if not others else ' On this sound the app also shows %s, which %s not there; the rows marked "everything shown" are what comes out when those are moved as well.' % (', '.join(o.upper() for o in others), 'is' if len(others) == 1 else 'are')
    rows1 = [row('a', 'As it came in', 'with the %s on it' % nm, f['wet']), row('a', 'Amount at far left', 'the %s taken out' % nm, f['out'])]
    rows2 = [row('b', 'Second sound, dry', second, f['b_dry']), row('b', 'With the %s found above' % nm, 'fader in the middle', f['b_onto'])]
    if others: rows1.append(row('a', 'Everything shown at far left', 'score %.0f' % sc['out_all'], f['out_all'])); rows2.append(row('b', 'Everything shown, faders in the middle', 'score %.0f' % sc['onto_all'], f['b_all']))
    rows1.append(row('ref', 'For comparison', 'the sound before the %s was added' % nm, f['dry'])); rows2.append(row('ref', 'For comparison', 'the second sound through the real effect', f['b_real']))
    more = ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(extras))
    meta = '%s %s Scores: taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.%s' % (E(real), E(found_text(group, d)), sc['out'], sc['onto'], E(note))
    return panel(title, meta, 'One sound: the %s group' % GROUP[group], rows1, 'Two sounds: the found %s on a second sound' % nm, rows2, more)
def wav(p): return sf.read(p, dtype='float64', always_2d=True)[0]

# ---------------------------------------------------------------- 18: a chorus of two voices (scoreboard case, an outside tool)
parts.append(short('chorus', ('chorus', 'cho_calf_01'), 'A', 'c1', '18. A drum machine pattern through a chorus',
                   'The real effect: two copies of the sound a side, 6 to 8.4 ms behind it, drifting 1.7 times a second, with the effect\'s amount at 60 %.', 'taiko hits',
                   [('Second sound, Speed at far right', 'the copies drift twice as fast', ['chorus.speed=2']), ('Second sound, Depth at far left', 'the copies stand still: a doubling without movement', ['chorus.depth=0']), ('Second sound, Amount at far right', 'the copies twice as loud', ['chorus.amount=2'])]))
# ---------------------------------------------------------------- 19: the copy louder than the sound (extra case x02 of Research/chorus/mkx.py)
X = '/dev/shm/chx/x02_'
parts.append(short('chorus', None, 'A', 'c2', '19. A chorus whose copy is louder than the sound',
                   'The real effect: one copy 7 to 13 ms behind the sound, drifting 0.8 times a second, at 2.3 times the sound\'s level (the effect\'s mix at 70 %). The app tells which of the two is the sound and takes the louder one out, working from the end of the sound backwards.', 'a sample loop',
                   [('Second sound, Amount at far right', 'the copy twice as loud again', ['chorus.amount=2']), ('Second sound, Speed at far left', 'the copy drifts half as fast', ['chorus.speed=0'])],
                   sounds=(wav(X + 'A_wet.wav'), wav(X + 'A_dry.wav'), wav(X + 'B_dry.wav'), wav(X + 'B_wet.wav'))))
# ---------------------------------------------------------------- 20: a long sound with a chorus (Research/chorus/mklong.py)
def long_chorus():
    L = '/dev/shm/chl/'; wet, dry, dryb, wetb = wav(L + 'longA_wet.wav'), wav(L + 'longA_dry.wav'), wav(L + 'longB_dry.wav'), wav(L + 'longB_wet.wav')
    f, sc, shown, d, full = render('chorus', 'c3', wet, dry, dryb, wetb, [], (8.0, 24.0)); info['c3'] = dict(scores=sc, shown=shown, found=d); others = [s for s in shown if s != 'chorus']
    note = '' if not others else ' On this sound the app also shows %s, which %s not there.' % (', '.join(o.upper() for o in others), 'is' if len(others) == 1 else 'are')
    meta = ('42 seconds of sound (seven short sounds laid end to end). The real effect: one copy 6 to 12 ms behind the sound, drifting 0.6 times a second, at two thirds of the sound\'s level, on all of it. %s '
            'A long sound is read in pieces of 12 seconds that are faded into one another. What you hear is the stretch from 0:08 to 0:24. Scores over the whole sound: taken out <b>%.0f</b>, put on a second sound of 27 seconds <b>%.0f</b>.%s'
            % (E(found_text('chorus', d)), sc['out'], sc['onto'], E(note)))
    rows1 = [row('a', 'As it came in', 'with the chorus on it', f['wet']), row('a', 'Amount at far left', 'the chorus taken out', f['out']), row('ref', 'For comparison', 'the sound before the chorus was added', f['dry'])]
    rows2 = [row('b', 'Second sound, dry', 'another 27 seconds of short sounds, the same stretch', f['b_dry']), row('b', 'With the chorus found above', 'fader in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])]
    return panel('20. A chorus on a long sound', meta, 'One sound: the CHORUS group', rows1, 'Two sounds: the found chorus on a second sound', rows2, '')
parts.append(long_chorus())
# ---------------------------------------------------------------- 21: a filter sweep (scoreboard case, a tool the model never saw)
parts.append(short('filtersweep', ('filtersweep', 'swp_mda_00'), 'A', 's1', '21. A sample loop through a filter sweep',
                   'The real effect: a low-pass filter with a peak at its edge that opens and closes once every 1.8 seconds.', 'an alto sax melody',
                   [('Second sound, Speed at far left', 'the filter moves half as fast', ['filtersweep.speed=0']), ('Second sound, Amount at far right', 'the filter twice as deep', ['filtersweep.amount=2'])]))
# ---------------------------------------------------------------- 22: a long sound where a filter sweep comes in half way (Research/moving/mklong.py)
def long_sweep():
    M = '/dev/shm/mvl/'; cs = {c['name']: c for c in json.load(open(M + 'list.json'))}; c = cs['swH01']; k = cs[c['dry']]; own = sum(p.startswith('real') for p in k['parts'])
    wet = wav(M + 'swH01_wet.wav'); dry = wav(M + c['dry'] + '.wav') * c['gain']; other = 'clean06'; ko = cs[other]; ownb = sum(q.startswith('real') for q in ko['parts']); dryb = wav(M + other + '.wav'); t0 = c['from_s']; cut = (t0 - 6.0, t0 + 10.0)
    f, sc, shown, d, full = render('filtersweep', 's2', wet, dry, dryb, None, [], cut, (min(cut[0], len(dryb) / SR - 16.0), min(cut[0], len(dryb) / SR - 16.0) + 16.0)); sp = spans_of(d['moving']); a = int(sp[0][0] * SR) if sp else 0
    same = float(np.abs(full['out'][:max(a - 4800, 0)] - wet[:max(a - 4800, 0)]).max()) if a > 4800 else None; h = int(t0 * SR); half = score.removed(dry[h:], wet[h:], full['out'][h:])[0]; info['s2'] = dict(scores=sc, half=half, shown=shown, found=d, before_max_diff=same, own=own)
    late = sp[0][0] - t0 if sp else 0.0; when = 'its start is found to within a tenth of a second' if abs(late) < 0.1 else ('the app picks it up %.1f seconds after it starts' % late if late > 0 else 'the app has it start %.1f seconds early' % -late)
    where = 'from %s to %s' % (clockt(sp[0][0]), clockt(sp[-1][1])) if sp else 'nowhere'; others = [s for s in shown if s != 'filtersweep']
    note = '' if not others else ' On this sound the app also shows %s, which %s not there.' % (', '.join(o.upper() for o in others), 'is' if len(others) == 1 else 'are')
    meta = ('%.0f seconds of sound (%s short sounds laid end to end%s). The real effect: the filter sweep of example 21 with other settings, switched on after %.0f seconds. '
            'The app found the filter and where it is: %s; %s. What you hear is the stretch from %s to %s. Before the filter comes in the sound comes back %s. '
            'Score for taking it out, over the part that has the filter: <b>%.0f</b>.%s'
            % (len(wet) / SR, {5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}.get(len(k['parts']), str(len(k['parts']))), '' if not own else (', one of them your own recording' if own == 1 else ', %d of them your own recordings' % own), t0, where, when, clockt(cut[0]), clockt(cut[1]),
               'exactly as it went in, sample for sample' if same is not None and same < 1e-9 else 'nearly as it went in', half, E(note)))
    rows1 = [row('a', 'As it came in', 'the filter comes in after 6 seconds of this stretch', f['wet']), row('a', 'Amount at far left', 'the filter taken off where it is, the rest left alone', f['out']), row('ref', 'For comparison', 'the sound before the filter was added', f['dry'])]
    rows2 = [row('b', 'Second sound, dry', 'another %.0f seconds of short sounds%s, a stretch of 16 seconds' % (len(dryb) / SR, '' if not ownb else (', one of them your own recording' if ownb == 1 else ', %s of them your own recordings' % {2: 'two', 3: 'three'}.get(ownb, str(ownb)))), f['b_dry']), row('b', 'With the filter found above', 'it comes in at the same time as on the first sound, %.0f seconds into this stretch' % (t0 - min(cut[0], len(dryb) / SR - 16.0)), f['b_onto'])]
    return panel('22. A filter sweep that comes in halfway', meta, 'One sound: the FILTER SWEEP group', rows1, 'Two sounds: the found filter on a second sound', rows2, '')
parts.append(long_sweep())
# ---------------------------------------------------------------- 23: a phaser (scoreboard case, a tool the model never saw; the case's second sound is the one the app is sure about)
parts.append(short('phaser', ('phaser', 'pha_pedalboard_01'), 'B', 'p1', '23. A bass loop through a phaser',
                   'The real effect: a phaser whose notches travel up and down once every 2.5 seconds, mixed half and half with the sound.', 'trumpet hits',
                   [('Second sound, Speed at far right', 'the notches travel twice as fast', ['phaser.speed=2']), ('Second sound, Amount at far right', 'the notches twice as deep', ['phaser.amount=2'])]))

head = open(os.path.join(HERE, 'new044_head.html')).read() if os.path.exists(os.path.join(HERE, 'new044_head.html')) else '<section id="new044"><h2>Chorus, phaser and filter sweep: new in build 0.4.4</h2></section>'
open(os.path.join(HERE, 'new044.html'), 'w').write(head + '\n' + '\n'.join(parts)); json.dump(info, open(os.path.join(HERE, 'new044_info.json'), 'w'), indent=1, default=float)
print('section written |', sum(os.path.getsize(os.path.join(AUD, n)) for n in os.listdir(AUD) if n.startswith('nw_')) // 1024, 'KB of audio')
