# The page's section for build 0.5.2 (the declipper: flat tops rebuilt at DISTORTION's far left): five examples rendered
# by the app's own processing (the command-line tool drrender, this build's and build 0.5.1's), as MP3s in
# audio/q2_*.mp3, and the examples' HTML in flat052_panels.html (build.py puts flat052_head.html and these last).
#   python3 flat052_section.py
# Picked by fixed rules from the result files of the trial (Research/declip/trial.py) and of the scoreboard:
#   a plain cut       of the outside hard clippers' cases whose tops are rebuilt and shown as DISTORTION: the middle one by score
#   soft flat tops    the scoreboard's cases that this build changes (there are two)
#   left alone        of the hard clippers' cases that the engine's self-test refuses: the one build 0.5.1 made worst
#   worse             of the 162 soft clippings whose tops are rebuilt: the one that loses most against build 0.5.1
import os, sys, re, json, html, glob, pickle, subprocess, numpy as np, soundfile as sf, scipy.signal as ss
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path[:0] = ['/home/claude/defect-refect/Research/bend', '/home/claude/defect-refect/Research/detect', '/home/claude/defect-refect/Research/declip']
import score, truth, build as B, bfeat, bengine
AUD = os.path.join(HERE, 'audio'); SR = 48000; TMP = '/tmp/claude-0/v2/flat052/'; os.makedirs(TMP, exist_ok=True); FIRST = int(os.environ.get('FIRST', '52')); E = html.escape
NEW = os.environ.get('EXE', '/tmp/claude-0/v2/declip/bin/drrender_052'); OLD = os.environ.get('EXE_OLD', '/tmp/claude-0/v2/four/build/drrender_new'); RES = '/home/claude/defect-refect/Resources'
SETS = '/tmp/claude-0/v2/bench/sets/'; SETS3 = '/tmp/claude-0/v2/declip/sets3/clip/'; MANY = os.environ.get('DR_DECLIP_MANY', '/dev/shm/dmany/'); DATA = '/tmp/claude-0/v2/declip/data/'; TRIAL = os.environ.get('TRIAL', 'r3')
BOARD = os.environ.get('BOARD', '/tmp/claude-0/v2/declip/fb052.jsonl'); BOARD_OLD = os.environ.get('BOARD_OLD', '/tmp/claude-0/v2/declip/fb051.jsonl')
SECOND = os.environ.get('SECOND', '/tmp/claude-0/v2/declip/second12.json')         # Research/declip/second.py: a found cut on a second sound, the twelve hard clippers' cases
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
BEND = ('distortion', 'saturation'); TEST_MIN = 20          # TEST_MIN as in Source/Core/Declip.h: under this score of its self-test the engine rebuilds nothing

def mp3(name, x, gain):
    wav = os.path.join(AUD, name + '.wav'); y = np.clip(x * gain, -1.0, 1.0); n = len(y); f = min(480, n // 4); y[:f] *= np.linspace(0, 1, f)[:, None]; y[-f:] *= np.linspace(1, 0, f)[:, None]
    sf.write(wav, y, SR, subtype='PCM_24'); subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame', '-b:a', '256k', os.path.join(AUD, name + '.mp3')], check=True); os.remove(wav)
    return 'audio/%s.mp3' % name
def row(cls, label, small, src): return '<div class="row %s"><div class="lab">%s<small>%s</small></div><audio controls preload="none" src="%s"></audio></div>' % (cls, E(label), E(small), src)
def run(exe, first, blocks):
    args = []
    for i, (second, out, fad) in enumerate(blocks): args += (['--'] if i else []) + [second, out] + list(fad)
    r = subprocess.run([exe, RES, first] + args, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
    return r.stdout
def rd(p, n): y, _ = sf.read(p, dtype='float64', always_2d=True); assert len(y) >= n, (p, len(y), n); return y[:n]
def val(s, k): m = re.search(r'(?:^| )%s=([0-9.eE+-]+)' % k, s); return float(m.group(1)) if m else None
def look(log):
    """What a run of drrender says: the groups shown, and the line of the bend engine."""
    return [s for s, _ in re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', log, re.M)], (re.search(r'^bend: (.*)$', log, re.M) or [None, ''])[1]
def render(tag, wetA, dryA, dryB=None, wetB=None):
    """The first sound alone with the group's Amount at far left (this build and build 0.5.1), half way left and at far right;
    with a second sound, that sound through what was found, the other groups off (this build and build 0.5.1)."""
    nA = len(wetA); pa = TMP + tag + '_a.wav'; sf.write(pa, wetA, SR, subtype='FLOAT'); S = dict(wet=wetA, dry=dryA); p = lambda k: TMP + tag + '_' + k + '.wav'
    own = lambda v: ['distortion.amount=%s' % v, 'saturation.amount=%s' % v]; others = [k + '.amount=0' for k in ALL if k not in BEND]
    bn = [('-', p('out'), own(0)), ('-', p('half'), own(0.5)), ('-', p('twice'), own(2))]; bo = [('-', p('out51'), own(0))]
    if dryB is not None: pb = TMP + tag + '_b.wav'; sf.write(pb, dryB, SR, subtype='FLOAT'); bn.append((pb, p('b_onto'), others)); bo.append((pb, p('b_onto51'), others)); S.update(b_dry=dryB, b_real=wetB)
    shown, line = look(run(NEW, pa, bn)); shown51, line51 = look(run(OLD, pa, bo))
    for _, f, _ in bn + bo: k = os.path.basename(f)[len(tag) + 1:-4]; S[k] = rd(f, len(dryB) if k.startswith('b_') else nA)
    sc = dict(out=score.removed(dryA, wetA, S['out'])[0], out51=score.removed(dryA, wetA, S['out51'])[0])
    if dryB is not None: sc.update(onto=score.removed(wetB, dryB, S['b_onto'])[0], onto51=score.removed(wetB, dryB, S['b_onto51'])[0])
    g = 0.89 / max(np.abs(v).max() for v in S.values()); f = {k: mp3('q2_%s_%s' % (tag, k), v, g) for k, v in S.items()}
    print(tag, 'shown:', shown, '| in 0.5.1:', shown51, '|', line[:200], '| 0.5.1:', line51[:60], '| scores', {k: round(v, 1) for k, v in sc.items()}, '| peaks', {k: round(float(np.abs(v).max()), 2) for k, v in S.items()}, flush=True)
    return f, sc, shown, shown51, line, S
def panel(title, meta, blocks, more):
    return ('<section class="panel"><h3>%s</h3><div class="meta">%s</div>%s%s</section>' % (E(title), meta, ''.join('<div class="sub">%s</div><div class="rows">%s</div>' % (E(sub), ''.join(rows)) for sub, rows in blocks),
            ('<details><summary>Other places of the fader on this example</summary><div class="rows">%s</div></details>' % ''.join(more)) if more else ''))
num = lambda v: ('%.0f' % v) if abs(v - round(v)) < 0.05 else ('%.1f' % v)
sco = lambda v: ('%.0f' % (round(v) + 0.0)).replace('-', '−')       # a score, with a real minus sign
pct = lambda v: ('%.0f' % (100 * v)) if 100 * v >= 9.95 else ('%.1f' % (100 * v))

def true_top(dry, wet):
    """By how many dB the real effect squeezed the loudest values, measured on the sound before and after it (as Research/bend/beval.py)."""
    x, y = dry, wet; cc = ss.correlate(y[:, 0] + y[:, 1], x[:, 0] + x[:, 1], 'full', 'fft'); n = len(x); k = np.arange(-4, 5) + n - 1; lag = int(k[np.argmax(np.abs(cc[k]))]) - (n - 1)
    pol = 1.0 if np.sum(x * np.roll(y, -lag, 0)) >= 0 else -1.0; T = np.clip(bfeat.answer(pol * np.roll(x, lag, 0), y), 0.0, 40.0)
    return max(bengine.tidy(T[0], bengine.SPREAD)[-1], bengine.tidy(T[1], bengine.SPREAD)[-1])

# ---------------------------------------------------------------- the result files, and the examples by their rules
def trial(part):
    rows = []
    for f in sorted(glob.glob(DATA + 'd38_%s_%s_?.pkl' % (TRIAL, part))): rows += pickle.load(open(f, 'rb'))
    return rows
T_all = trial('all'); T_many = trial('many'); assert len([r for r in T_all if not r['name'].startswith('x_')]) == 84 and len(T_many) == 162, (len(T_all), len(T_many))
hardbox = [r for r in T_all if r['name'].startswith(('clip_clip_sox_', 'clip_clip_pedalboard_', 'clip_clip_ff_hard_'))]; assert len(hardbox) == 12 and all('hard' in r['kinds'] for r in hardbox)
rebuilt = sorted((r for r in hardbox if r['rebuilt'] and r['found_new']), key=lambda r: -r['new']); t_cut = rebuilt[len(rebuilt) // 2]
refused = [r for r in hardbox if r['did'] and not r['rebuilt'] and r['found_new'] and r['test'] < TEST_MIN]; t_alone = min(refused, key=lambda r: r['plain'])
soft_rebuilt = [r for r in T_many if r.get('rebuilt')]; t_worse = min(soft_rebuilt, key=lambda r: r['new'] - r['plain'])
rows = lambda f: {r['name']: r for r in (json.loads(l) for l in open(f) if l.strip())}
new, old = rows(BOARD), rows(BOARD_OLD); assert len(new) == 136 and sorted(new) == sorted(old)
changed = sorted(n for n in new if abs((new[n]['out'] or 0) - (old[n]['out'] or 0)) >= 0.1 or abs(new[n]['onto'] - old[n]['onto']) >= 0.1 or bool(new[n]['found']) != bool(old[n]['found']))        # a tenth of a point: the level rule moves one case by 0.07
assert changed == ['dist_pedalboard_00', 'dist_pedalboard_01'], changed
print('plain cut:', t_cut['name'], '(of %d rebuilt and shown)' % len(rebuilt), '| left alone:', t_alone['name'], '(of %d refused)' % len(refused), '| worse:', t_worse['name'], '(of %d rebuilt)' % len(soft_rebuilt), '| scoreboard:', changed, flush=True)
by_trial = {r['name']: r for r in T_all + T_many}

def many_source(name):
    """The corpus sound a case of the 162 was made from: the maker's draws again (Research/declip/mk_many.py; it stores no names), checked against its list."""
    os.environ.setdefault('DR_SETS', SETS); os.environ.setdefault('DR_CORPUS', '/tmp/claude-0/v2/corpus/'); argv = sys.argv; sys.argv = sys.argv[:1]
    import mkpos, clip_fx
    sys.argv = argv; rng = np.random.default_rng(5); names = sorted(mkpos.meta); L = {r['name']: r for r in json.load(open(MANY + 'list.json'))}; got = {}
    for eff, o in [('clip_ff_' + k, o) for k in clip_fx.KINDS if k != 'hard' for o in (1, 2, 4)] + [('dist_pedalboard', 1)]:
        for i in range(6 if eff != 'dist_pedalboard' else 18):
            n = str(rng.choice(names)); d = float(rng.uniform(14, 34)) if eff == 'dist_pedalboard' else float(rng.uniform(8, 28)); got['m_%s_%d_%02d' % (eff.replace('clip_ff_', '').replace('dist_', ''), o, i)] = (n, d)
    assert len(got) == len(L) == 162 and all(abs(got[k][1] - L[k]['drive']) < 1e-12 and ('own' if 'real' in got[k][0] else got[k][0].split('_')[0]) == L[k]['kind'] for k in L)
    return got[name][0], L[name]

parts = []; info = {}; N = [FIRST]
def add(tag, html_, **kw): info[tag] = kw; parts.append(html_); N[0] += 1
WAY51 = 'the bend walked back along the curve that was read; tops that were flat stay flat'
WAY52 = 'the tops rebuilt; no value comes out higher than the highest was, so the sound is quieter'
def close(a, b, tol=1.5): return abs(a - b) < tol

# ---------------------------------------------------------------- a plain cut
def cut_panel(t, tag):
    d = SETS3 + t['name'][5:] + '/'; cA, cB = truth.load(d, 'A'), truth.load(d, 'B'); m = cA['meta']; na, nb = B.sound_name(m['A']), B.sound_name(m['B']); assert m['effect'] == 'clip_sox'
    f, sc, shown, shown51, line, S = render(tag, cA['wet'], cA['dry'], cB['dry'], cB['wet']); g51 = [s for s in shown51 if s in BEND]; assert 'distortion' in shown and len(g51) == 1 and 'tops=cut/cut' in line and val(line, 'rebuilt') == 1
    assert close(sc['out'], t['new']) and close(sc['out51'], t['plain']), (sc, t['new'], t['plain'])          # the page's renders are the trial's
    top, drive, test = val(line, 'top_db'), m['params']['drive_db'], val(line, 'test'); assert float(np.abs(S['out']).max()) <= float(np.abs(S['wet']).max()) * (1 + 1e-9)
    two = json.load(open(SECOND)); mine = two[t['name'][5:]]; assert close(mine['new']['onto'], sc['onto'], 0.5) and close(mine['old']['onto'], sc['onto51'], 0.5), (mine, sc)
    six = sorted(v['new']['onto'] for v in two.values() if v['new']['rebuilt']); six51 = [v['old']['onto'] for v in two.values() if v['new']['rebuilt']]; assert len(six) == 6 and len(two) == 12 and min(six) > 0
    meta = ('%s. The real effect: a plain overload. The sound was turned up by %s dB against full scale, so its loudest tops stood %s dB above the ceiling, and what did not fit was cut off: %s %% of all its values. '
            'The app found the cut, rebuilt the tops from the sound around them, and reads that the loudest of them stood %s dB above the cut. '
            'Before it rebuilds a sound, the app tests itself on that sound: it cuts tops that are whole and rebuilds those. Here the test scored %s of 100, and from %d on the tops are rebuilt. '
            'Scores for taking the effect out: build 0.5.1 <b>%s</b>, this build <b>%s</b>.%s '
            'The cut goes onto a second sound by how high the rebuilt tops stand: that sound is pushed as far past the cut. Here the height was read almost to the tenth of a dB, and the second sound scores <b>%s</b> (build 0.5.1: %s). '
            'On the six test cases of this kind it scores between %s and %s, %s on average (build 0.5.1: %s).%s'
            % (E(B.cap(na)), num(drive), num(drive), pct(t['share']), num(top), sco(test), TEST_MIN, sco(sc['out51']), sco(sc['out']), ' Build 0.5.1 showed this sound as SATURATION.' if g51 == ['saturation'] else '',
               sco(sc['onto']), sco(sc['onto51']), sco(six[0]), sco(six[-1]), sco(np.mean(six)), sco(np.mean(six51)), E(also(shown))))
    assert abs(top - drive) < 0.25                                                                             # 'almost to the tenth of a dB'
    add(tag, panel('%d. A sound cut off flat' % N[0], meta,
        [('One sound: the DISTORTION group', [row('a', 'As it came in', 'cut off flat', f['wet']), row('a', 'Amount at far left in build 0.5.1', WAY51, f['out51']),
                                              row('a', 'Amount at far left in build 0.5.2', WAY52, f['out']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])]),
         ('Two sounds: the found cut on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'Through the cut found above', 'fader in the middle: its loudest values as far past the cut as the first sound\'s rebuilt tops stand', f['b_onto']),
                                                          row('b', 'The same in build 0.5.1', 'through the curve that build read off the first sound', f['b_onto51']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])])],
        [row('a', 'Amount half way left', 'the rebuilt sound pushed into the same cut again, 12 dB less hard than it was', f['half']), row('a', 'Amount at far right', 'the rebuilt sound pushed %s dB harder into the cut than it was' % num(top), f['twice'])]),
        case=t['name'], scores=sc, shown=shown, shown51=shown51, top=top, drive=drive, test=test, share=t['share'], six=six, six51=six51)
def also(shown):
    others = [s for s in shown if s not in BEND]; assert all(o == 'reverb' for o in others), others      # 'not there' is known of the REVERB the app shows on some of these cases; any other group needs a look first
    return '' if not others else ' On this sound the app also shows REVERB, which is not there.'
cut_panel(t_cut, 'c')

# ---------------------------------------------------------------- soft flat tops: the two scoreboard cases this build changes
def soft_panel(name, tag, title):
    r = new[name]; r0 = old[name]; t = by_trial['board_' + name]; d = SETS + r['fam'] + '/' + name + '/'; cA, cB = truth.load(d, 'A'), truth.load(d, 'B'); m = cA['meta']; na, nb = B.sound_name(m['A']), B.sound_name(m['B']); assert m['effect'] == 'dist_pedalboard'
    f, sc, shown, shown51, line, S = render(tag, cA['wet'], cA['dry'], cB['dry'], cB['wet']); assert 'distortion' in shown and 'distortion' in shown51 and val(line, 'rebuilt') == 1 and 'soft' in line.split('tops=')[1].split()[0]
    assert abs(sc['out'] - r['out']) < 0.5 and abs(sc['onto'] - r['onto']) < 0.5 and abs(sc['out51'] - r0['out']) < 0.5 and abs(sc['onto51'] - r0['onto']) < 0.5, (sc, r['out'], r['onto'], r0['out'], r0['onto'])      # the page's renders are the scoreboard's
    assert abs(sc['onto'] - sc['onto51']) < 0.05 and float(np.abs(S['b_onto'] - S['b_onto51']).max()) < 1e-6      # 'as in build 0.5.1'
    top, real, second = val(line, 'top_db'), true_top(cA['dry'], cA['wet']), val(line, 'second') == 1
    passes = ('It rebuilt them from the sound around them, read the curve off what it had rebuilt, and with the curve known rebuilt them a second time, which is what far left gives.' if second else
              'It rebuilt them from the sound around them and read the curve off what it had rebuilt. A second pass with the curve known came out too rough to be trusted, so far left gives the first pass.')
    meta = ('%s. The real effect: a plain distortion, one curve and nothing else, driven by %s dB. It does not cut the tops off, it presses them nearly flat: %s %% of all values sit in such tops. '
            'Measured on the sound before and after the effect, the loudest values were squeezed by %s dB; the app reads %s dB. The app found the pressed tops. %s '
            'Scores for taking the effect out: build 0.5.1 <b>%s</b>, this build <b>%s</b>. On a second sound the curve is put on as in build 0.5.1 (%s).%s'
            % (E(B.cap(na)), num(m['params']['drive_db']), pct(t['share']), num(real), num(top), passes, sco(sc['out51']), sco(sc['out']), sco(sc['onto']), E(also(shown))))
    add(tag, panel('%d. %s' % (N[0], title), meta,
        [('One sound: the DISTORTION group', [row('a', 'As it came in', 'through the effect', f['wet']), row('a', 'Amount at far left in build 0.5.1', WAY51, f['out51']),
                                              row('a', 'Amount at far left in build 0.5.2', WAY52, f['out']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])]),
         ('Two sounds: the found curve on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'Through the curve found above', 'fader in the middle, as in build 0.5.1', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])])],
        [row('a', 'Amount half way left', 'the rebuilt sound pushed into the same curve again, 12 dB less hard than it was', f['half']), row('a', 'Amount at far right', 'the rebuilt sound pushed %s dB harder into the curve than it was' % num(top), f['twice'])]),
        case=name, scores=sc, shown=shown, top=top, true_top=real, second=second, share=t['share'])
    os.remove(os.path.join(AUD, 'q2_%s_b_onto51.mp3' % tag))          # the same sound as b_onto: not on the page
soft_panel(changed[0], 's1', 'A distortion that presses the tops flat')
soft_panel(changed[1], 's2', 'The same distortion on another sound')

# ---------------------------------------------------------------- a dense sound: the self-test says no
def alone_panel(t, tag):
    d = SETS3 + t['name'][5:] + '/'; cA = truth.load(d, 'A'); m = cA['meta']; na = B.sound_name(m['A']); assert m['effect'] in ('clip_ff_hard', 'clip_pedalboard', 'clip_sox')
    f, sc, shown, shown51, line, S = render(tag, cA['wet'], cA['dry']); assert 'distortion' in shown and 'distortion' in shown51 and 'tops=cut/cut' in line and val(line, 'rebuilt') == 0
    assert close(sc['out51'], t['plain'], 3.0) and float(np.abs(S['out'] - S['wet']).max()) < 1e-6 and float(np.abs(S['half'] - S['wet']).max()) < 1e-6, (sc, t['plain'])      # 'does nothing left of the middle'
    test = val(line, 'test'); assert test < TEST_MIN
    box = dict(clip_ff_hard='a hard clipper', clip_pedalboard='a clipper plugin', clip_sox='a plain overload')[m['effect']]
    meta = ('%s. The real effect: %s, driven by %s dB, that cuts %s %% of all values off flat. The app found the cut and tested itself on this sound: it cut tops that are whole and rebuilt those, and scored %s of 100. '
            'That is under the %d it asks for, because this sound is dense and what a top was cannot be told from the sound around it. So nothing is rebuilt, and left of the middle the fader leaves the sound as it is. '
            'Build 0.5.1 walked a bend back here that cannot be walked back that way. Scores for taking the effect out: build 0.5.1 <b>%s</b>, this build <b>%s</b>.%s'
            % (E(B.cap(na)), box, num(m['params']['drive_db']), pct(t['share']), sco(test), TEST_MIN, sco(sc['out51']), sco(sc['out']), E(also(shown))))
    add(tag, panel('%d. A dense sound that is left alone' % N[0], meta,
        [('One sound: the DISTORTION group', [row('a', 'As it came in', 'cut off flat', f['wet']), row('a', 'Amount at far left in build 0.5.1', WAY51, f['out51']),
                                              row('a', 'Amount at far left in build 0.5.2', 'nothing rebuilt: the sound as it came in', f['out']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])])], []),
        case=t['name'], scores=sc, shown=shown, test=test, share=t['share'])
    for k in ('half', 'twice'): os.remove(os.path.join(AUD, 'q2_%s_%s.mp3' % (tag, k)))
alone_panel(t_alone, 'a')

# ---------------------------------------------------------------- the rebuilt case that loses most against build 0.5.1
def worse_panel(t, tag):
    src, l = many_source(t['name']); wet = sf.read(MANY + t['name'] + '_wet.wav', dtype='float64', always_2d=True)[0]; dry = sf.read(MANY + t['name'] + '_dry.wav', dtype='float64', always_2d=True)[0]
    assert l['effect'].startswith('clip_ff_') and l['over'] in (1, 2, 4)
    f, sc, shown, shown51, line, S = render(tag, wet, dry); assert val(line, 'rebuilt') == 1 and close(sc['out'], t['new'], 3.0) and close(sc['out51'], t['plain'], 3.0), (sc, t['new'], t['plain'])
    group = [s for s in shown if s in BEND]; group51 = [s for s in shown51 if s in BEND]; assert len(group) == 1 and len(group51) == 1, (shown, shown51)
    was = '' if group == group51 else ' Build 0.5.1 showed this sound as %s.' % group51[0].upper()
    rate = {1: 'at the file\'s own sample rate', 2: 'at twice the sample rate inside', 4: 'at four times the sample rate inside'}[l['over']]; real, top = true_top(dry, wet), val(line, 'top_db')
    meta = ('%s. The real effect: a soft clipper that works %s, driven by %s dB. Such a clipper leaves tops that are flat with a ripple on them, which is harder to read than a curve\'s work. '
            'Measured on the sound before and after the effect, the loudest values were squeezed by %s dB; the app reads %s dB. It rebuilds the tops, and its self-test gave it %s of 100. '
            'The result lies further from the sound before the effect than build 0.5.1\'s. Scores for taking the effect out: build 0.5.1 <b>%s</b>, this build <b>%s</b>.%s '
            'Of the %d cases of the trial whose tops are rebuilt this is the one that loses most. The effect changed this sound only a little, so the score swings widely for a small difference: your ears say more here than the number.%s'
            % (E(B.cap(B.sound_name(src))), rate, num(l['drive']), num(real), num(top), sco(val(line, 'test')), sco(sc['out51']), sco(sc['out']), was, len(soft_rebuilt), E(also(shown))))
    add(tag, panel('%d. The case that loses most against build 0.5.1' % N[0], meta,
        [('One sound: the %s group' % group[0].upper(), [row('a', 'As it came in', 'through the effect', f['wet']), row('a', 'Amount at far left in build 0.5.1', WAY51, f['out51']),
                                                         row('a', 'Amount at far left in build 0.5.2', WAY52, f['out']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])])], []),
        case=t['name'], scores=sc, shown=shown, top=top, true_top=real, test=val(line, 'test'), over=l['over'], drive=l['drive'])
    for k in ('half', 'twice'): os.remove(os.path.join(AUD, 'q2_%s_%s.mp3' % (tag, k)))
worse_panel(t_worse, 'w')

open(os.path.join(HERE, 'flat052_panels.html'), 'w').write('\n'.join(parts)); json.dump(info, open(os.path.join(HERE, 'flat052_info.json'), 'w'), indent=1, default=float)
print('panels written |', sum(os.path.getsize(os.path.join(AUD, n)) for n in os.listdir(AUD) if n.startswith('q2_')) // 1024, 'KB of audio')
