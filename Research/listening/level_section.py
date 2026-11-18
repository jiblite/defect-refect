# The page's section for build 0.4.3 (level movement and flanger): six examples rendered by the app's own processing (the
# command-line tool drrender), as MP3s in audio/lv_*.mp3, and the section's HTML in level.html (build.py puts it after the echo
# section).
#   python3 level_section.py
import os, sys, re, json, html, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); import score, truth
HERE = os.path.dirname(os.path.abspath(__file__)); AUD = os.path.join(HERE, 'audio'); SR = 48000; TMP = '/dev/shm/lp/'; os.makedirs(TMP, exist_ok=True)
EXE = '/tmp/claude-0/drbuild/drrender'; RES = '/home/claude/defect-refect/Resources'; SETS = '/tmp/claude-0/v2/bench/sets/'
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
E = html.escape
def mp3(name, x, gain):
    wav = os.path.join(AUD, name + '.wav'); y = np.clip(x * gain, -1.0, 1.0); n = len(y); f = min(480, n // 4); y[:f] *= np.linspace(0, 1, f)[:, None]; y[-f:] *= np.linspace(1, 0, f)[:, None]
    sf.write(wav, y, SR, subtype='PCM_24'); subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame', '-b:a', '256k', os.path.join(AUD, name + '.mp3')], check=True); os.remove(wav)
    return 'audio/%s.mp3' % name
def row(cls, label, small, src): return '<div class="row %s"><div class="lab">%s<small>%s</small></div><audio controls preload="none" src="%s"></audio></div>' % (cls, E(label), E(small), src)
def run(first, blocks):
    """One analysis of the first sound, several renders: blocks = (second or '-', out, [fader=value ..]). Returns what the app printed."""
    args = []
    for i, (second, out, fad) in enumerate(blocks): args += (['--'] if i else []) + [second, out] + list(fad)
    r = subprocess.run([EXE, RES, first] + args, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
    return r.stdout
def rd(p, n): y, _ = sf.read(p, dtype='float64', always_2d=True); return y[:n]
def render(group, tag, wetA, dryA, dryB, wetB, extras, cut=None):
    """The app on the first sound (wetA) alone and with the second (dryB). cut = (from, to) in s: only that stretch goes to the page."""
    nA, nB = len(wetA), len(dryB); pa, pb = TMP + tag + '_a.wav', TMP + tag + '_b.wav'; sf.write(pa, wetA, SR, subtype='FLOAT'); sf.write(pb, dryB, SR, subtype='FLOAT')
    others = [k + '.amount=0' for k in ALL if k != group]
    blocks = [('-', TMP + tag + '_o0.wav', [group + '.amount=0']), ('-', TMP + tag + '_o1.wav', [k + '.amount=0' for k in ALL]), (pb, TMP + tag + '_o2.wav', others), (pb, TMP + tag + '_o3.wav', [])]
    for i, (_, _, fad) in enumerate(extras): blocks.append((pb, TMP + tag + '_x%d.wav' % i, others + fad))
    log = run(pa, blocks); shown = re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', log, re.M)
    x = dict(wet=wetA, out=rd(blocks[0][1], nA), out_all=rd(blocks[1][1], nA), dry=dryA, b_dry=dryB, b_onto=rd(blocks[2][1], nB), b_all=rd(blocks[3][1], nB), b_real=wetB)
    for i in range(len(extras)): x['x%d' % i] = rd(TMP + tag + '_x%d.wav' % i, nB)
    sc = dict(out=score.removed(dryA, wetA, x['out'])[0], out_all=score.removed(dryA, wetA, x['out_all'])[0], onto=score.removed(wetB, dryB, x['b_onto'])[0], onto_all=score.removed(wetB, dryB, x['b_all'])[0])
    full = dict(x)
    if cut is not None: a, b = int(cut[0] * SR), int(cut[1] * SR); x = {k: v[a:b] for k, v in x.items()}
    g = 0.89 / max(np.abs(v).max() for v in x.values()); f = {k: mp3('lv_%s_%s' % (tag, k), v, g) for k, v in x.items()}
    lev = re.search(r'^level: (.*)$', log, re.M); flg = re.search(r'^flanger: (.*)$', log, re.M)
    print(tag, 'shown:', shown, '|', lev.group(0) if lev else None, '|', flg.group(0) if flg else None, '| scores', {k: round(v, 1) for k, v in sc.items()}, flush=True)
    return f, sc, [s for s, _ in shown], dict(re.findall(r'([a-z_]+)=([^ ]+)', (lev.group(1) if group != 'flanger' else flg.group(1)) if (lev and flg) else '')), full
def panel(title, meta, one, rows1, two, rows2, more):
    return ('<section class="panel"><h3>%s</h3><div class="meta">%s</div><div class="sub">%s</div><div class="rows">%s</div><div class="sub">%s</div><div class="rows">%s</div>%s</section>'
            % (E(title), meta, E(one), ''.join(rows1), E(two), ''.join(rows2), ('<details><summary>The other faders on this example</summary><div class="rows">%s</div></details>' % more) if more else ''))
NAME = dict(tremolo='tremolo', autopan='auto pan', gatedrhythm='gate', flanger='flanger'); GROUP = dict(tremolo='TREMOLO', autopan='AUTO PAN', gatedrhythm='GATED RHYTHM', flanger='FLANGER')
parts = []; info = {}

# ---------------------------------------------------------------- four cases of 6 s from the scoreboard: of each kind the middle one of those that are shown
SHORT = [('tremolo', 'trm_calf_01', 'tr', '12. A full band with a tremolo', 'The real effect: the level goes down and back 3.9 times a second.', 'one of your recordings',
          [('Second sound, Speed at far right', 'the same movement twice as fast', ['tremolo.speed=2']), ('Second sound, Amount at far right', 'twice as deep', ['tremolo.amount=2'])]),
         ('autopan', 'pan_sox_01', 'pn', '13. Computer speech with an auto pan', 'The real effect: the sound moves from side to side 0.7 times a second, nearly all the way.', 'a drum loop',
          [('Second sound, Speed at far left', 'the same movement at half the speed', ['autopan.speed=0'])]),
         ('gatedrhythm', 'gat_sox_01', 'gt', '14. Electric piano chords through a gate', 'The real effect: a gate that opens 3.2 times a second and is open for a little over half of the time.', 'bursts of noise',
          [('Second sound, Speed at far right', 'the gate twice as fast', ['gatedrhythm.speed=2'])]),
         ('flanger', 'flg_swh_00', 'fl', '16. A drum kit through a flanger', 'The real effect: a copy of the sound 2.4 to 8 ms behind it, its delay moving once in four seconds, fed back a little.', 'computer speech',
          [('Second sound, Speed at far right', 'the sweep twice as fast', ['flanger.speed=2']), ('Second sound, Feedback at far right', 'the comb rings more', ['flanger.feedback=2']), ('Second sound, Amount at far right', 'the copy twice as loud', ['flanger.amount=2'])])]
def found_text(group, d):
    if group == 'flanger':
        lo, hi = d.get('delay_ms', '0..0').split('..'); return 'The app found a copy at %.0f %% of the sound\'s level, its delay moving between %.1f and %.1f ms.' % (100 * abs(float(d.get('gain', 0))), float(lo), float(hi))
    r, dep = float(d.get('rate_hz', 0)), float(d.get('depth_db', 0))
    if group == 'autopan': return 'The app found a movement from side to side %.2f times a second, up to %.0f dB between the sides.' % (r, dep)
    if group == 'gatedrhythm': return 'The app found a gate that opens %.2f times a second and takes the sound down by %.0f dB when closed.' % (r, dep)
    return 'The app found a level that goes down by %.1f dB and back %.2f times a second.' % (dep, r)
def short(group, case, tag, title, real, second, extras):
    fam = group; cA = truth.load(SETS + fam + '/' + case + '/', 'A'); cB = truth.load(SETS + fam + '/' + case + '/', 'B')
    f, sc, shown, d, _ = render(group, tag, cA['wet'], cA['dry'], cB['dry'], cB['wet'], extras); info[case] = dict(scores=sc, shown=shown, found=d)
    others = [s for s in shown if s != group]; nm = NAME[group]
    note = '' if not others else ' On this sound the app also shows %s, which %s not there; the rows marked "everything shown" are what comes out when those are moved as well.' % (', '.join(o.upper() for o in others), 'is' if len(others) == 1 else 'are')
    rows1 = [row('a', 'As it came in', 'with the %s on it' % nm, f['wet']), row('a', 'Amount at far left', 'the %s taken out' % nm, f['out'])]
    rows2 = [row('b', 'Second sound, dry', second, f['b_dry']), row('b', 'With the %s found above' % nm, 'fader in the middle', f['b_onto'])]
    if others: rows1.append(row('a', 'Everything shown at far left', 'score %.0f' % sc['out_all'], f['out_all'])); rows2.append(row('b', 'Everything shown, faders in the middle', 'score %.0f' % sc['onto_all'], f['b_all']))
    rows1.append(row('ref', 'For comparison', 'the sound before the %s was added' % nm, f['dry'])); rows2.append(row('ref', 'For comparison', 'the second sound through the real effect', f['b_real']))
    more = ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(extras))
    meta = '%s %s Scores: taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.%s' % (E(real), E(found_text(group, d)), sc['out'], sc['onto'], E(note))
    return panel(title, meta, 'One sound: the %s group' % GROUP[group], rows1, 'Two sounds: the found %s on a second sound' % nm, rows2, more)

# ---------------------------------------------------------------- two long sounds with the effect on a part only
def clockt(s): s = int(round(s)); return '%d:%02d' % (s // 60, s % 60)
def spans_of(d): return [tuple(float(v) for v in p.split('..')) for p in d.get('there_s', '').split(',') if '..' in p]
def long_gate():
    """Example 15: one of the long test sounds of Research/level/long.py (part 38): a gate on the second half of a minute of sound."""
    p = '/dev/shm/lc/part_38_'
    if not os.path.exists(p + 'wet.wav'): subprocess.run(['python3', 'mk_case.py', 'part', '38'], cwd='/tmp/claude-0/v2/level', check=True)
    wet, dry, dryb, wetb = (sf.read(p + t + '.wav', dtype='float64', always_2d=True)[0] for t in ('wet', 'dry', 'dryb', 'wetb')); cut = (26.0, 46.0)
    f, sc, shown, d, full = render('gatedrhythm', 'lg', wet, dry, dryb, wetb, [], cut); sp = spans_of(d); a = int(sp[0][0] * SR) if sp else 0
    same = float(np.abs(full['out'][:max(a - 4800, 0)] - wet[:max(a - 4800, 0)]).max()) if a > 4800 else None; info['long_gate'] = dict(scores=sc, shown=shown, found=d, before_max_diff=same)
    where = 'from %s to %s' % (clockt(sp[0][0]), clockt(sp[0][1])) if len(sp) == 1 else 'in %d stretches' % len(sp)
    meta = ('A minute of sound (eleven short sounds laid end to end, two of them your own recordings). The real effect: a gate that opens 5.6 times a second, switched on after 34 seconds. '
            'The app found the gate and where it is: %s. What you hear is the stretch from 0:26 to 0:46. Before the gate comes in the sound comes back %s. Scores over the whole minute: taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.'
            % (where, 'exactly as it went in, sample for sample' if same is not None and same < 1e-9 else 'nearly as it went in', sc['out'], sc['onto']))
    rows1 = [row('a', 'As it came in', 'the gate comes in after 8 seconds of this stretch', f['wet']), row('a', 'Amount at far left', 'the gate taken out where it is, the rest left alone', f['out']), row('ref', 'For comparison', 'the sound before the gate was added', f['dry'])]
    rows2 = [row('b', 'Second sound, dry', 'another minute of short sounds, the same stretch', f['b_dry']), row('b', 'With the gate found above', 'it comes in at the same time as on the first sound', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])]
    return panel('15. A gate that comes in halfway', meta, 'One sound: the GATED RHYTHM group', rows1, 'Two sounds: the found gate on a second sound', rows2, '')
def long_flanger():
    """Example 17: one of the long test sounds of Research/sweep/flong.py (part 4): a flanger that comes in after 20 s."""
    p = '/dev/shm/fl/part_4_'
    if not os.path.exists(p + 'a.wav'): subprocess.run(['python3', 'fmk.py', 'part', '4'], cwd='/tmp/claude-0/v2/sweep', check=True)
    wet, dry, dryb, wetb = (sf.read(p + t + '.wav', dtype='float64', always_2d=True)[0] for t in ('a', 'd', 'b', 't')); cut = (14.0, 34.0)
    f, sc, shown, d, full = render('flanger', 'lf', wet, dry, dryb, wetb, [], cut); sp = spans_of(d); a = int(sp[0][0] * SR) if sp else 0
    late = (sp[0][0] - json.load(open(p + 'info.json'))['live'][0]) if sp else 0.0      # how long after the flanger's real start the app's stretch begins
    when = 'its start is found to within a tenth of a second' if abs(late) < 0.1 else ('the app picks it up %.1f seconds after it starts' % late if late > 0 else 'the app has it start %.1f seconds early' % -late)
    same = float(np.abs(full['out'][:max(a - 4800, 0)] - wet[:max(a - 4800, 0)]).max()) if a > 4800 else None; info['long_flanger'] = dict(scores=sc, shown=shown, found=d, before_max_diff=same)
    where = 'from %s to %s' % (clockt(sp[0][0]), clockt(sp[0][1])) if len(sp) == 1 else 'in %d stretches' % len(sp)
    others = [s for s in shown if s != 'flanger']
    note = '' if not others else ' On this sound the app also shows %s, which %s not there.' % (', '.join(o.upper() for o in others), 'is' if len(others) == 1 else 'are')
    meta = ('47 seconds of sound (nine short sounds laid end to end, two of them your own recordings). The real effect: a flanger switched on after 20 seconds. '
            'The app found the copy and where it is: %s; %s. What you hear is the stretch from 0:14 to 0:34. Before the flanger the sound comes back %s. '
            'Score for taking it out, over the whole sound: <b>%.0f</b>. Onto a second sound a flanger goes all the way, so there is no fair score for that here.%s'
            % (where, when, 'exactly as it went in, sample for sample' if same is not None and same < 1e-9 else 'nearly as it went in', sc['out'], E(note)))
    rows1 = [row('a', 'As it came in', 'the flanger comes in after 6 seconds of this stretch', f['wet']), row('a', 'Amount at far left', 'the flanger taken out where it is, the rest left alone', f['out']), row('ref', 'For comparison', 'the sound before the flanger was added', f['dry'])]
    rows2 = [row('b', 'Second sound, dry', 'another 47 seconds of short sounds, the same stretch', f['b_dry']), row('b', 'With the flanger found above', 'on the whole of the second sound', f['b_onto'])]
    return panel('17. A flanger that comes in after 20 seconds', meta, 'One sound: the FLANGER group', rows1, 'Two sounds: the found flanger on a second sound', rows2, '')

for i, ex in enumerate(SHORT):
    parts.append(short(*ex))
    if i == 2: parts.append(long_gate())
parts.append(long_flanger())
head = open(os.path.join(HERE, 'level_head.html')).read() if os.path.exists(os.path.join(HERE, 'level_head.html')) else '<section id="level"><h2>Level movement and flanger: new in build 0.4.3</h2></section>'
open(os.path.join(HERE, 'level.html'), 'w').write(head + '\n' + '\n'.join(parts)); json.dump(info, open(os.path.join(HERE, 'level_info.json'), 'w'), indent=1)
print('level section written |', sum(os.path.getsize(os.path.join(AUD, n)) for n in os.listdir(AUD) if n.startswith('lv_')) // 1024, 'KB of audio')
