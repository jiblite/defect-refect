# The page's section for build 0.5.0 (the MOVEMENT group: Pitch, Loudness, Tone, Stereo, Spectral): eight examples rendered by
# the app's own processing (the command-line tool drrender), as MP3s in audio/m5_*.mp3, and the examples' HTML in
# move050_panels.html (build.py puts move050_head.html and these after the section of build 0.4.7).
#   python3 move050_section.py
# The examples are the owner's own recordings, picked by fixed rules (see PICK) from what the app's code reads in each
# of the 40 (survey.json, made by the container-only survey.py with the app's reader). CONTAINER ONLY: the recordings
# are named by their index in the sorted list of ids here and in everything this writes or prints, never by an id.
import os, sys, re, json, html, subprocess, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path[:0] = ['/tmp/claude-0/v2/proto', '/tmp/claude-0/v2/bench']
import truth
AUD = os.path.join(HERE, 'audio'); SR = 48000; TMP = '/tmp/claude-0/v2/move050/'; os.makedirs(TMP, exist_ok=True); FIRST = int(os.environ.get('FIRST', '40'))
EXE = os.environ.get('EXE', '/tmp/claude-0/v2/mv/cc/drrender_v9'); RES = '/home/claude/defect-refect/Resources'; C = '/tmp/claude-0/v2/corpus/'; SV = '/tmp/claude-0/v2/mv/page/sv/'
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
MV = ['pitch', 'loudness', 'tone', 'stereo', 'spectral']; OFF = [k + '.amount=0' for k in ALL]
meta = json.load(open(C + 'meta.json')); real = sorted(k for k, v in meta.items() if v.get('kind') == 'real-recording'); path = lambda i: C + 'dry/' + real[i] + '.wav'
E = html.escape

# ---------------------------------------------------------------- the rules
rows_ = json.load(open('/tmp/claude-0/v2/mv/page/survey.json')); S = []
for r in rows_:                                         # a recording that is in the set twice counts once (the first of them)
    key = (round(r['cents'], 3), round(r['loud'], 4), round(r['tone'], 4), round(r['level'], 4))
    if key not in [k for k, _ in S]: S.append((key, r))
S = [r for _, r in S]
well = sorted((r for r in S if r['held'] >= 0.8), key=lambda r: r['cents'])                 # the pitch followed over four fifths of the sound or more, by how far it moves
PICK = dict(
    wide=well[-1]['i'],                                                                      # the widest movement of those
    mid=well[len(well) // 2]['i'],                                                           # the middle one of those
    part=max((r for r in S if r['held'] < 0.5), key=lambda r: r['cents'])['i'],              # the widest movement among the recordings followed for under half of the sound
    loud=max(S, key=lambda r: r['loud_range'])['i'],                                         # the widest level movement (between the 5th and the 95th percentile of the level)
    tone=max(S, key=lambda r: r['tone_slow'])['i'],                                          # the widest slow movement of the tone
    pan=max((r for r in S if not r['mono']), key=lambda r: r['pan_range'])['i'],             # the widest movement between the sides
    steady=min((r for r in S if r['held'] >= 0.95), key=lambda r: r['cents'])['i'])          # followed throughout and moving least: the second sound of the two-sound example
print('recordings counted once:', len(S), '| picked (by index):', PICK, flush=True)
by = {r['i']: r for r in rows_}

# ---------------------------------------------------------------- rendering
def mp3(name, x, gain):
    wav = os.path.join(AUD, name + '.wav'); y = np.clip(x * gain, -1.0, 1.0); n = len(y); f = min(480, n // 4); y[:f] *= np.linspace(0, 1, f)[:, None]; y[-f:] *= np.linspace(1, 0, f)[:, None]
    sf.write(wav, y, SR, subtype='PCM_24'); subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame', '-b:a', '256k', os.path.join(AUD, name + '.mp3')], check=True); os.remove(wav)
    return 'audio/%s.mp3' % name
def row(cls, label, small, src): return '<div class="row %s"><div class="lab">%s<small>%s</small></div><audio controls preload="none" src="%s"></audio></div>' % (cls, E(label), E(small), src)
def rd(p): return sf.read(p, dtype='float64', always_2d=True)[0]
def run(first, blocks):
    args = []
    for i, (second, out, fad) in enumerate(blocks): args += (['--'] if i else []) + [second, out] + list(fad)
    r = subprocess.run([EXE, RES, first] + args, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
    return r.stdout
def render(tag, first, second, sets):
    """sets: (key, [fader=value ...]). With a second sound the found effects of the first are left off: only MOVEMENT works."""
    blocks = [(second or '-', TMP + '%s_%s.wav' % (tag, k), (OFF if second else []) + fad) for k, fad in sets]; log = run(first, blocks)
    Sd = {'in': rd(first)}
    if second: Sd['b_in'] = rd(second)
    n = len(Sd['b_in' if second else 'in'])
    for (k, _), (_, out, _) in zip(sets, blocks): y = rd(out); assert abs(len(y) - n) <= 1, (tag, k, len(y), n); Sd[k] = y
    g = 0.89 / max(np.abs(v).max() for v in Sd.values()); f = {k: mp3('m5_%s_%s' % (tag, k), v, g) for k, v in Sd.items()}
    mv = re.findall(r'^movement of the (first|second) sound: pitch followed on ([0-9.eE+-]+) % over ([0-9.eE+-]+) cents, level ([0-9.eE+-]+) dB, (mono|stereo)', log, re.M)
    shown = re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', log, re.M); base = re.findall(r'^movement: the first sound is measured without (.*)$', log, re.M)
    print(tag, '| movement:', mv, '| shown:', [s for s, _ in shown], '| measured without:', base, '| peak before gain %.2f' % (0.89 / g), flush=True)
    return f, {w: dict(held=float(h), cents=float(c), level=float(l), mono=m == 'mono') for w, h, c, l, m in mv}, Sd

# ---------------------------------------------------------------- the pitch curve as the app read it
def ticks(lo, hi):
    for step in (25, 50, 100, 200, 300, 600, 1200):
        a, b = np.floor(lo / step) * step, np.ceil(hi / step) * step
        if (b - a) / step <= 4: return a, b, step
    return np.floor(lo / 1200) * 1200, np.ceil(hi / 1200) * 1200, 1200
def curve(i, label):
    """The curve the app read in recording i (sv/r%02d.own.pitch.f64, written by survey.py with the app's reader): cents around the sound's average."""
    v = np.fromfile(SV + 'r%02d.own.pitch.f64' % i); T = len(v) // 2; c = v[:T] - v[:T].mean(); h = v[T:] > 0.5
    c = np.round(c).astype(int); lo, hi, step = ticks(min(c.min(), -25), max(c.max(), 25)); W, Hh = 600.0, 100.0
    X = lambda t: W * t / (T - 1); Y = lambda q: Hh * (hi - q) / (hi - lo); pc = lambda a: '%.2f%%' % a; out = []
    out.append('<figure class="curve"><figcaption>%s</figcaption><div class="plot">' % E(label))
    out.append('<svg viewBox="0 0 600 100" preserveAspectRatio="none" tabindex="0" role="img" aria-label="%s" data-c="%s" data-h="%s" data-lo="%d" data-hi="%d">'
               % (E(label + ' Left and right arrow keys step through it.'), ','.join(str(int(q)) for q in c), ''.join('1' if q else '0' for q in h), lo, hi))
    bands = []; t = 0
    while t < T:
        if h[t]: t += 1; continue
        e = t
        while e < T and not h[e]: e += 1
        bands.append((t, e)); t = e
    for a, b in bands: out.append('<rect class="none" x="%.1f" y="0" width="%.1f" height="100"/>' % (X(max(a - 0.5, 0)), X(min(b - 0.5, T - 1)) - X(max(a - 0.5, 0))))
    q = lo
    while q <= hi + 1e-9: out.append('<line class="grid%s" x1="0" x2="600" y1="%.2f" y2="%.2f" vector-effect="non-scaling-stroke"/>' % (' zero' if abs(q) < 1e-9 else '', Y(q), Y(q))); q += step
    out.append('<polyline class="line" vector-effect="non-scaling-stroke" points="%s"/></svg>' % ' '.join('%.1f,%.2f' % (X(t), Y(c[t])) for t in range(T)))
    q = lo
    while q <= hi + 1e-9: out.append('<span class="yt" style="top:%s">%s</span>' % (pc(100 * (hi - q) / (hi - lo)), ('%+d' % q) if q else '0')); q += step
    secs = int((T - 1) / 100)
    for s_ in range(secs + 1): out.append('<span class="xt" style="left:%s">%d s</span>' % (pc(100 * s_ * 100 / (T - 1)), s_))
    for a, b in bands:
        if (b - a) / (T - 1) >= 0.18: out.append('<span class="nr" style="left:%s;width:%s">not read</span>' % (pc(100 * a / (T - 1)), pc(100 * (b - a) / (T - 1))))
    out.append('<i class="hair"></i><i class="dot"></i><div class="tip" role="status"></div></div>')
    trs = ''.join('<tr><td class="n">%.1f s</td><td class="n">%s</td><td>%s</td></tr>' % (t / 100.0, ('%+d' % c[t]) if c[t] else '0', 'read' if h[t] else 'not read, the curve stands still') for t in range(0, T, 50))
    out.append('<details><summary>The curve in numbers</summary><div class="tablewrap"><table><tr><th class="n">Time</th><th class="n">Cents</th><th></th></tr>%s</table></div></details></figure>' % trs)
    return ''.join(out), dict(lo=int(c.min()), hi=int(c.max()), read=float(h.mean()))
def panel(title, meta_, fig, blocks_):
    return '<section class="panel"><h3>%s</h3><div class="meta">%s</div>%s%s</section>' % (E(title), meta_, fig, ''.join('<div class="sub">%s</div><div class="rows">%s</div>' % (E(s), ''.join(r)) for s, r in blocks_))
def wide_words(cents):
    o = cents / 1200.0
    return 'about %s' % ('a semitone' if cents < 150 else '%d semitones' % round(cents / 100.0)) if cents < 1100 else 'an octave' if o < 1.2 else 'an octave and a half' if 1.35 <= o < 1.65 else 'two octaves' if 1.85 <= o < 2.15 else '%.1f octaves' % o
pct = lambda v: '%.0f %%' % (100 * v)
parts = []; info = {}; N = [FIRST]
def add(tag, html_, **kw): info[tag] = dict(kw, number=N[0]); parts.append(html_); N[0] += 1
CAP = 'What the app read in this recording: its pitch against its own average, in cents (100 cents are a semitone). Grey: not read, the curve stands still there.'

# ---------------------------------------------------------------- Pitch, one sound
def pitch_example(tag, i, title, lead, half):
    sets = [('flat', ['movement.pitch=0'])] + ([('half', ['movement.pitch=0.5'])] if half else []) + [('twice', ['movement.pitch=2'])]
    f, mv, _ = render(tag, path(i), None, sets); m = mv['first']; fig, cv = curve(i, CAP)
    assert abs(m['held'] / 100 - cv['read']) < 0.005 and abs(m['cents'] - (cv['hi'] - cv['lo'])) < 2.5, (tag, m, cv)          # the curve drawn is the one the render used
    meta_ = '%s The app follows the pitch over <b>%s</b> of the sound, and what it follows moves over <b>%.0f cents</b>, %s.' % (lead, pct(m['held'] / 100), m['cents'], wide_words(m['cents']))
    rows = [row('a', 'As it came in', 'one of your recordings', f['in']), row('a', 'Pitch at far left', 'the movement that was read is taken out', f['flat'])]
    if half: rows.append(row('a', 'Pitch half way left', 'half of it is left', f['half']))
    rows.append(row('a', 'Pitch at far right', 'the movement that was read, twice', f['twice']))
    add(tag, panel('%d. %s' % (N[0], title), meta_, fig, [('One sound: the Pitch fader', rows)]), rec=i, movement=m, curve=cv)
pitch_example('pw', PICK['wide'], 'Pitch: the recording of yours whose pitch moves furthest', 'Of your recordings in which the pitch is followed over four fifths of the sound or more, this is the one that moves furthest.', True)
pitch_example('pm', PICK['mid'], 'Pitch: a small movement', 'Of the same recordings, this is the middle one by how far the pitch moves.', False)
pitch_example('pp', PICK['part'], 'Pitch: a recording that is read only in part', 'Of your recordings in which the pitch is followed over less than half of the sound, this is the one with the widest movement. It shows the limit: where the reader cannot follow, the curve stands still, and what moves there stays in the sound at every setting of the fader.', False)

# ---------------------------------------------------------------- Loudness, Tone and Spectral, Stereo: one sound each
i = PICK['loud']; f, mv, _ = render('lo', path(i), None, [('flat', ['movement.loudness=0']), ('twice', ['movement.loudness=2'])])
meta_ = 'Of your recordings this is the one whose level moves furthest: <b>%.0f dB</b> lie between its quiet and its loud moments (the 5th and the 95th percentile of the level).' % by[i]['loud_range']
add('lo', panel('%d. Loudness: the recording of yours whose level moves furthest' % N[0], meta_, '', [('One sound: the Loudness fader',
    [row('a', 'As it came in', 'one of your recordings', f['in']), row('a', 'Loudness at far left', 'the level held even', f['flat']), row('a', 'Loudness at far right', 'the level movement, twice', f['twice'])])]), rec=i, movement=mv.get('first'))
i = PICK['tone']; f, mv, _ = render('to', path(i), None, [('flat', ['movement.tone=0']), ('twice', ['movement.tone=2']), ('neutral', ['movement.spectral=0']), ('colour', ['movement.spectral=2'])])
meta_ = 'Of your recordings this is the one whose tone moves furthest over time: the balance between its upper and its lower half shifts by <b>%.0f dB</b>. Tone is that shift over time. Spectral is the colour the sound has on average.' % by[i]['tone_slow']
add('to', panel('%d. Tone and Spectral: the recording of yours whose tone moves furthest' % N[0], meta_, '', [('One sound: the Tone fader',
    [row('a', 'As it came in', 'one of your recordings', f['in']), row('a', 'Tone at far left', 'the tone held still, the level left as it is', f['flat']), row('a', 'Tone at far right', 'the tone movement, twice', f['twice'])]),
    ('One sound: the Spectral fader', [row('a', 'Spectral at far left', 'neutral: the same power in every octave', f['neutral']), row('a', 'Spectral at far right', "the sound's own colour, once more", f['colour'])])]), rec=i, movement=mv.get('first'))
i = PICK['pan']; f, mv, _ = render('st', path(i), None, [('flat', ['movement.stereo=0']), ('twice', ['movement.stereo=2'])])
meta_ = 'Of your stereo recordings this is the one that moves furthest between the sides: by <b>%.0f dB</b> from one end of its way to the other.' % by[i]['pan_range']
add('st', panel('%d. Stereo: the recording of yours that moves furthest between the sides' % N[0], meta_, '', [('One sound: the Stereo fader',
    [row('a', 'As it came in', 'one of your recordings', f['in']), row('a', 'Stereo at far left', 'held at its average place and width', f['flat']), row('a', 'Stereo at far right', 'the movement, twice', f['twice'])])]), rec=i, movement=mv.get('first'))

# ---------------------------------------------------------------- two sounds: the first sound's movement on a second
rest = ['movement.%s=2' % k for k in MV if k != 'pitch']; five = ['movement.%s=2' % k for k in MV]
def onto(tag, second, title, lead, second_name):
    f, mv, _ = render(tag, path(PICK['wide']), second, [('pitch', ['movement.pitch=2']), ('rest', rest), ('all', five), ('flat', ['movement.pitch=0'])]); m2 = mv['second']
    meta_ = '%s The second sound keeps its length, and its own pitch movement (followed over %s of it, %.0f cents wide) gives way to the first sound\'s. The found effects of the first sound are switched off here, so that only MOVEMENT works.' % (lead, pct(m2['held'] / 100), m2['cents'])
    add(tag, panel('%d. %s' % (N[0], title), meta_, '', [('Two sounds: MOVEMENT works on the second sound',
        [row('a', 'First sound', 'the recording of example %d' % FIRST, f['in']), row('b', 'Second sound as it came in', second_name, f['b_in']),
         row('b', 'Pitch at far right', "the first sound's pitch movement in place of its own", f['pitch']), row('b', 'The other four at far right', "the first sound's loudness, tone, stereo and colour, the pitch left alone", f['rest']),
         row('b', 'All five at far right', 'everything that moves in the first sound', f['all']), row('b', 'Pitch at far left', "the second sound's own pitch movement taken out", f['flat'])])]), second=second_name, movement=mv)
onto('ts', path(PICK['steady']), 'Two sounds: the movement of example %d on a steady recording of yours' % FIRST,
     'The second sound is the recording of yours whose pitch is followed throughout and moves least.', 'another of your recordings')
pad = truth.load('/tmp/claude-0/v2/bench/sets/tone/tone_fil4_04/', 'B'); assert pad['meta']['B'] == 'fs_chd_warmpad'; sf.write(TMP + 'pad.wav', pad['dry'], SR, subtype='FLOAT')
onto('tp', TMP + 'pad.wav', 'Two sounds: the same movement on pad chords', 'The second sound is not a recording of yours: warm pad chords from the test material, a sound with nothing of an engine in it.', 'warm pad chords')

open(os.path.join(HERE, 'move050_panels.html'), 'w').write('\n'.join(parts) + '\n'); json.dump(dict(pick=PICK, examples=info, counted=len(S)), open(os.path.join(HERE, 'move050_info.json'), 'w'), indent=1)
print('examples %d to %d written' % (FIRST, N[0] - 1))
