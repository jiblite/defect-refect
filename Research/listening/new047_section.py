# The page's section for build 0.4.7 (Haas width, stereo width, ring modulation, bitcrush, stutter): seven examples
# rendered by the app's own processing (the command-line tool drrender), as MP3s in audio/n7_*.mp3, and the examples'
# HTML in new047_panels.html (build.py puts new047_head.html and these after the section of build 0.4.6).
#   python3 new047_section.py
# Six examples are scoreboard cases picked by fixed rules from the scoreboard's result file (see pick): for each group
# the middle case of those on which it is shown, ranked by its score; of the two cases with the crusher that mixes
# plain sound in, the weaker one. The seventh is a file at 44.1 kHz with a crusher and a stutter of our own on it
# (Research/moving/mknew.py makes it), because both engines read such a file at its own rate.
import os, sys, re, json, html, subprocess, numpy as np, soundfile as sf
from scipy.signal import resample_poly
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score, truth, build as B
AUD = os.path.join(HERE, 'audio'); SR = 48000; TMP = '/tmp/claude-0/v2/new047/'; os.makedirs(TMP, exist_ok=True); FIRST = int(os.environ.get('FIRST', '33'))
EXE = os.environ.get('EXE', '/tmp/claude-0/drrender047g'); RES = '/home/claude/defect-refect/Resources'; SETS = '/tmp/claude-0/v2/bench/sets/'; BOARD = os.environ.get('BOARD', '/tmp/claude-0/v2/bench/famboard_047f.jsonl')
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
def ended(x, keep=0.25):
    """Without the silence at its end (a stutter's far left closes the sound up; the app's file keeps its length)."""
    live = np.where(np.abs(x).max(1) > 1e-5)[0]; return x[:min(len(x), (int(live[-1]) if len(live) else 0) + int(keep * SR))]
def render(group, tag, wetA, dryA, dryB, wetB, own, extras, first=None, rate=SR):
    """The app on the first sound alone and with the second. own: (key, [fader=value ...]) renders of the first sound alone; extras: the same with the second sound, the other groups off."""
    nA, nB = (len(wetA) if first is None else int(round(len(wetA) * SR / rate))), len(dryB); pa, pb = first or (TMP + tag + '_a.wav'), TMP + tag + '_b.wav'
    if first is None: sf.write(pa, wetA, SR, subtype='FLOAT')
    sf.write(pb, dryB, SR, subtype='FLOAT'); others = [k + '.amount=0' for k in ALL if k != group] if group else []
    blocks = [('-', TMP + tag + '_o1.wav', [k + '.amount=0' for k in ALL]), (pb, TMP + tag + '_o2.wav', others), (pb, TMP + tag + '_o3.wav', [])]
    blocks += [('-', TMP + tag + '_y%d.wav' % i, fad) for i, (_, fad) in enumerate(own)] + [(pb, TMP + tag + '_x%d.wav' % i, others + fad) for i, (_, _, fad) in enumerate(extras)]
    log = run(pa, blocks); shown = re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', log, re.M); wet48 = wetA if rate == SR else resample_poly(wetA, 160, 147, axis=0)[:nA]
    S = dict(wet=wet48, out_all=rd(blocks[0][1], nA), b_dry=dryB, b_onto=rd(blocks[1][1], nB), b_all=rd(blocks[2][1], nB))
    if dryA is not None: S['dry'] = dryA
    if not group: del S['b_onto']
    if wetB is not None: S['b_real'] = wetB
    for i, (key, _) in enumerate(own): S[key] = rd(TMP + tag + '_y%d.wav' % i, nA)
    for i in range(len(extras)): S['x%d' % i] = rd(TMP + tag + '_x%d.wav' % i, nB)
    sc = {}
    if dryA is not None and 'out' in S: sc['out'] = score.removed(dryA, wet48, S['out'])[0]
    if wetB is not None: sc['onto_all'] = score.removed(wetB, dryB, S['b_all'])[0]
    if wetB is not None and 'b_onto' in S: sc['onto'] = score.removed(wetB, dryB, S['b_onto'])[0]
    del S['out_all']                                    # not played on the page; with one group asked for, neither is "everything shown on the second sound"
    if group: del S['b_all']
    g = 0.89 / max(np.abs(v).max() for v in S.values())
    if 'closed' in S: S['closed'] = ended(S['closed'])
    f = {k: mp3('n7_%s_%s' % (tag, k), v, g) for k, v in S.items()}
    lines = {k: (re.search(r'^%s: (.*)$' % k, log, re.M) or [None, ''])[1] for k in ('sides', 'ring', 'crush', 'stutter')}
    print(tag, 'shown:', shown, '|', {k: v[:150] for k, v in lines.items() if 'found=0' not in v[:8]}, '| scores', {k: round(v, 1) for k, v in sc.items()}, flush=True)
    return f, sc, [s for s, _ in shown], lines, S
def panel(title, meta, one, rows1, two, rows2, more):
    return ('<section class="panel"><h3>%s</h3><div class="meta">%s</div><div class="sub">%s</div><div class="rows">%s</div><div class="sub">%s</div><div class="rows">%s</div>%s</section>'
            % (E(title), meta, E(one), ''.join(rows1), E(two), ''.join(rows2), ('<details><summary>The other faders on this example</summary><div class="rows">%s</div></details>' % more) if more else ''))
num = lambda v: ('%.0f' % v) if abs(v - round(v)) < 0.05 else ('%.1f' % v)
def side_text(s):
    """What the sides engine read, in words (the line drrender prints)."""
    if 'copy share=' in s and ' L ' not in s: return 'The app found a filtered copy of the middle at the sides.'
    m = re.search(r' L (.*?) R (.*?) there_s', s); L = re.findall(r'([0-9.eE+-]+)@([0-9.eE+-]+)ms', m.group(1)); R = re.findall(r'([0-9.eE+-]+)@([0-9.eE+-]+)ms', m.group(2))
    if len(L) == 1 and len(R) == 1:
        (gl, dl), (gr, dr) = (map(float, L[0]), map(float, R[0])); late, d, g = ('right', dr - dl, gr / gl) if dr > dl else ('left', dl - dr, gl / gr)
        return 'The app found the %s side %s ms behind the %s one, at %.0f %% of its level.' % (late, num(d), 'left' if late == 'right' else 'right', 100 * abs(g))
    reach = max(float(d) for _, d in L + R); late = 'right' if max(float(d) for _, d in R) > max(float(d) for _, d in L) else 'left'
    if 'found=haas' in s: return 'The app found a copy of the sound on the %s side, %s ms behind the direct sound%s.' % (late, num(reach), ' and turned upside down' if any(float(g) < 0 and float(d) > 1 for g, d in (R if late == 'right' else L)) else '')
    return 'The app found one sound reaching both sides, with copies of itself up to %s ms later that differ between the sides.' % num(reach)
def ring_text(s):
    hz = float(re.search(r' hz=([0-9.eE+-]+)', s).group(1)); rho = float(re.search(r'rho=([0-9.eE+-]+)', s).group(1)); part = 20 * np.log10(max(rho, 1e-6) / np.sqrt(0.5))
    return 'The app found a tone at %.1f Hz%s.' % (hz, ' and none of the plain sound left beside it' if rho < 0.02 else ', with plain sound left beside the modulated part, %s' % ('as loud as that part' if abs(part) < 0.5 else '%.0f dB %s it' % (abs(part), 'under' if part < 0 else 'over')))
def crush_text(s, rate=SR):
    """rate: the file's own rate (drrender prints the hold in samples at 48 kHz)."""
    kind = re.search(r'kind=(\w+)', s).group(1); bits = float(re.search(r'bits=([0-9.eE+-]+)', s).group(1)); st = float(re.search(r'step=([0-9.eE+-]+)', s).group(1)); hold = float(re.search(r'hold=([0-9.eE+-]+)', s).group(1)); mix = float(re.search(r'mix=([0-9.eE+-]+)', s).group(1))
    lev = '%.0f levels (%s bits)' % (2.0 / st, num(bits)); h = '' if hold <= 1.0 else ', each value held for %s samples (%.1f kHz)' % (num(hold * rate / SR), 48.0 / hold)
    if kind == 'soft': return 'The app found a crusher that mixes plain sound back in: %s%s, %.0f %% crushed.' % (lev, h, 100 * mix)
    return 'The app found the sound rounded to %s%s%s.' % (lev, h, ', with something after the crusher that it does not read' if kind == 'through' else '')
def events_of(s):
    """(start s, end s, slice ms) of each event drrender printed (the first 12)."""
    sp = [tuple(float(v) for v in q.split('..')) for q in re.search(r'at_s=([0-9.,eE+-]+)', s).group(1).split(',')]; sl = [float(v) for v in re.search(r'each_ms=([0-9.,eE+-]+)', s).group(1).split(',')]
    return [(a, b, L) for (a, b), L in zip(sp, sl)]
def stutter_text(s, secs):
    """secs: the sound's length (an event that runs into the file's end has fewer plays than it was given)."""
    n = int(re.search(r'events=(\d+)', s).group(1)); ev = events_of(s); sl = [L for _, _, L in ev]; fades = re.search(r'fades=(\d)', s).group(1) == '1'
    cut = len(ev) > 1 and ev[-1][1] > secs - 0.002; pl = [round((b - a) * 1000.0 / L, 1) for a, b, L in (ev[:-1] if cut else ev)]
    a = '%s ms' % num(sl[0]) if max(sl) - min(sl) < 0.5 else '%s to %s ms' % (num(min(sl)), num(max(sl))); p = '%s times' % num(pl[0]) if max(pl) - min(pl) < 0.05 else '%s to %s times' % (num(min(pl)), num(max(pl)))
    return 'The app found %d slice%s of %s, each played %s%s%s.' % (n, '' if n == 1 else 's', a, p, ' (the last one is cut off by the end of the file)' if cut else '', ', every play fading in and out' if fades else '')
def cut_of(s):
    """Seconds that a stutter's far left cuts out: every event's stretch but its first play."""
    return sum((b - a) - L / 1000.0 for a, b, L in events_of(s))
rows = [json.loads(l) for l in open(BOARD)]
def pick(fam, key, flt=lambda r: True, weakest=False):
    """The middle case (by its score, best first) of those of a family on which its own group is shown; weakest: the last."""
    R = sorted([r for r in rows if r['fam'] == fam and r['found'] and flt(r)], key=lambda r: -r[key]); return R[-1] if weakest else R[len(R) // 2]
# what the outside effect did, in plain words, for the cases the rules can pick
REAL = dict(
    haas_mda_00='The real effect: the sound is sent to both sides, and one side also gets a copy that comes 17 ms late and upside down.',
    haas_sox_00='The real effect: the right side is delayed by 6.6 ms and a little quieter.', haas_sox_01='The real effect: one side is delayed by a few milliseconds and a little quieter.',
    haas_ffmpeg_00='The real effect: one side is delayed by a few milliseconds.', haas_ffmpeg_01='The real effect: the right side is delayed by 14.8 ms and a little quieter.',
    wid_calf_haasenh_01='The real effect: the middle of the sound is copied to the sides twice, 1.0 and 7.5 ms late, the later copy upside down, both at 71 % of its level.',
    wid_calf_haasenh_00='The real effect: the middle of the sound is copied to the sides twice, a few milliseconds late.',
    wid_caps_00='The real effect: a widener that sends the middle to the sides through a short filter.', wid_caps_01='The real effect: a widener that sends the middle to the sides through a short filter.',
    wid_mda_comb_00='The real effect: a comb filter that differs between the two sides.', wid_mda_comb_01='The real effect: a comb filter that differs between the two sides.',
    bit_calf_01='The real effect: a crusher with about a thousand levels that holds every value for 4 samples, rounds its steps off a little and mixes 3 % of the plain sound back in.',
    bit_calf_00='The real effect: a crusher that holds its values, rounds its steps off and mixes some of the plain sound back in.',
    bit_swh_00='The real effect: the sound rounded to 16 levels, every value held for 8 samples.', bit_swh_01='The real effect: the sound rounded to a few levels, every value held for several samples.',
    bit_pedalboard_00='The real effect: the sound rounded to a few bits.', bit_pedalboard_01='The real effect: the sound rounded to a few bits.')
def real_text(r):
    p = truth.load(SETS + r['fam'] + '/' + r['name'] + '/', 'A')['meta']['params']
    if r['fam'] == 'ringmod': return 'The real effect: the sound multiplied by a tone at %.1f Hz, mixed %.0f to %.0f with the plain sound.' % (p['carrier'], 100 * p['mix'], 100 - 100 * p['mix']) if 'mix' in p else 'The real effect: the sound multiplied by a tone at %.1f Hz.' % p.get('carrier', p.get('freq', 0.0))
    if r['fam'] == 'stutter': return 'The real effect: slices of %.0f ms, each played %d times, at places along the sound chosen by chance.' % (p['slice_ms'], p['repeats'])
    return REAL[r['name']]
def case(r, side='A'):
    d = SETS + r['fam'] + '/' + r['name'] + '/'; cA = truth.load(d, side); cB = truth.load(d, 'B' if side == 'A' else 'A'); m = cA['meta']
    return cA['wet'], cA['dry'], cB['dry'], cB['wet'], B.sound_name(m[side]), B.sound_name(m['B' if side == 'A' else 'A'])
parts = []; info = {}; N = [FIRST]
def also(shown, group, sc):
    others = [s for s in shown if s != group]
    return '' if not others else ' On this sound the app also shows %s, which %s not there.' % (', '.join(o.upper() for o in others), 'is' if len(others) == 1 else 'are')
def add(tag, title, html_, **kw): info[tag] = kw; parts.append(html_); N[0] += 1

# ---------------------------------------------------------------- Haas width
r = pick('haas', 'out'); wetA, dryA, dryB, wetB, na, nb = case(r)
f, sc, shown, ln, _ = render('haas', 'h', wetA, dryA, dryB, wetB, [('out', ['haas.amount=0']), ('half', ['haas.amount=0.5'])], [('Second sound, Amount at far right', 'the late side later still, the two levels further apart', ['haas.amount=2'])])
meta = '%s. %s %s Scores: taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.%s' % (E(B.cap(na)), E(real_text(r)), E(side_text(ln['sides'])), sc['out'], sc['onto'], E(also(shown, 'haas', sc)))
add('h', '', panel('%d. A sound made wide with a delay on one side' % N[0], meta, 'One sound: the HAAS WIDTH group',
    [row('a', 'As it came in', 'one side late', f['wet']), row('a', 'Amount at far left', 'the delay taken out', f['out']), row('a', 'Amount half way left', 'half of it', f['half']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])],
    'Two sounds: the found delay on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'With the delay found above', 'fader in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])],
    row('b', 'Second sound, Amount at far right', 'the late side later still, the two levels further apart', f['x0'])), case=r['name'], scores=sc, shown=shown)
# ---------------------------------------------------------------- stereo width
r = pick('width', 'out'); wetA, dryA, dryB, wetB, na, nb = case(r)
f, sc, shown, ln, _ = render('width', 'w', wetA, dryA, dryB, wetB, [('out', ['width.amount=0']), ('twice', ['width.amount=2'])], [('Second sound, Amount at far right', 'the copies at the sides twice as loud', ['width.amount=2'])])
meta = '%s. %s %s Scores: taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.%s' % (E(B.cap(na)), E(real_text(r)), E(side_text(ln['sides'])), sc['out'], sc['onto'], E(also(shown, 'width', sc)))
add('w', '', panel('%d. A sound made wide from its middle' % N[0], meta, 'One sound: the STEREO WIDTH group',
    [row('a', 'As it came in', 'widened', f['wet']), row('a', 'Amount at far left', 'the widening taken out', f['out']), row('a', 'Amount at far right', 'twice as wide', f['twice']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])],
    'Two sounds: the found widening on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'With the widening found above', 'fader in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])],
    row('b', 'Second sound, Amount at far right', 'the copies at the sides twice as loud', f['x0'])), case=r['name'], scores=sc, shown=shown)
# ---------------------------------------------------------------- ring modulation
r = pick('ringmod', 'out'); wetA, dryA, dryB, wetB, na, nb = case(r)
ex = [('Second sound, Carrier at far left', 'the tone an octave lower', ['ringmod.carrier=0']), ('Second sound, Carrier at far right', 'the tone an octave higher', ['ringmod.carrier=2']), ('Second sound, Amount at far right', 'the tone part twice as strong', ['ringmod.amount=2'])]
f, sc, shown, ln, _ = render('ringmod', 'r', wetA, dryA, dryB, wetB, [('out', ['ringmod.amount=0']), ('half', ['ringmod.amount=0.5'])], ex)
meta = '%s. %s %s Scores: taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.%s' % (E(B.cap(na)), E(real_text(r)), E(ring_text(ln['ring'])), sc['out'], sc['onto'], E(also(shown, 'ringmod', sc)))
add('r', '', panel('%d. A sound through a ring modulator' % N[0], meta, 'One sound: the RING MOD group',
    [row('a', 'As it came in', 'multiplied by the tone', f['wet']), row('a', 'Amount at far left', 'the tone taken out', f['out']), row('a', 'Amount half way left', 'half of it', f['half']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])],
    'Two sounds: the found tone on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'With the tone found above', 'fader in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])],
    ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(ex))), case=r['name'], scores=sc, shown=shown)
# ---------------------------------------------------------------- the crusher that mixes plain sound in (it can be undone)
r = pick('bitcrush', 'out', lambda q: q['name'].startswith('bit_calf'), True); wetA, dryA, dryB, wetB, na, nb = case(r)
ex = [('Second sound, Bit Depth three quarters right', 'the steps four times as large', ['bitcrush.bits=1.5']), ('Second sound, Sample Rate at far right', 'every value held four times as long', ['bitcrush.down=2']), ('Second sound, Amount half way left', 'half way between the plain sound and the crushed one', ['bitcrush.amount=0.5'])]
f, sc, shown, ln, _ = render('bitcrush', 'cs', wetA, dryA, dryB, wetB, [('out', ['bitcrush.amount=0']), ('coarse', ['bitcrush.bits=1.5'])], ex)
meta = '%s. %s %s Scores: taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.%s' % (E(B.cap(na)), E(real_text(r)), E(crush_text(ln['crush'])), sc['out'], sc['onto'], E(also(shown, 'bitcrush', sc)))
add('cs', '', panel('%d. A crusher that can be undone' % N[0], meta, 'One sound: the BITCRUSH group',
    [row('a', 'As it came in', 'crushed', f['wet']), row('a', 'Amount at far left', 'the sound before the crusher, worked out from the crushed one', f['out']), row('a', 'Bit Depth three quarters right', 'the same sound through steps four times as large', f['coarse']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])],
    'Two sounds: the found crusher on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'Through the crusher found above', 'fader in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])],
    ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(ex))), case=r['name'], scores=sc, shown=shown)
# ---------------------------------------------------------------- a bare crusher (it cannot be undone)
r = pick('bitcrush', 'onto', lambda q: not q['name'].startswith('bit_calf')); wetA, dryA, dryB, wetB, na, nb = case(r)
ex = [('Second sound, Bit Depth half way left', 'the steps a quarter as large (two bits more)', ['bitcrush.bits=0.5']), ('Second sound, Sample Rate at far left', 'every value held a quarter as long', ['bitcrush.down=0']), ('Second sound, Amount half way left', 'half way between the plain sound and the crushed one', ['bitcrush.amount=0.5'])]
f, sc, shown, ln, _ = render('bitcrush', 'cb', wetA, dryA, dryB, wetB, [('twice', ['bitcrush.amount=2'])], ex)
meta = ('%s. %s %s What the rounding and the hold threw away is gone, so this crusher cannot be taken out: left of the middle the Amount fader does nothing. Score for putting it on the second sound: <b>%.0f</b>.%s'
        % (E(B.cap(na)), E(real_text(r)), E(crush_text(ln['crush'])), sc['onto'], E(also(shown, 'bitcrush', sc))))
add('cb', '', panel('%d. A crusher that cannot be undone' % N[0], meta, 'One sound: the BITCRUSH group',
    [row('a', 'As it came in', 'crushed', f['wet']), row('a', 'Amount at far right', 'through steps twice as large and a hold twice as long', f['twice']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])],
    'Two sounds: the found crusher on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'Through the crusher found above', 'fader in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])],
    ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(ex))), case=r['name'], scores=sc, shown=shown)
# ---------------------------------------------------------------- stutter
r = pick('stutter', 'onto'); wetA, dryA, dryB, wetB, na, nb = case(r)
ex = [('Second sound, Slice at far left', 'the slices half as long, played over the same stretch', ['stutter.slice=0']), ('Second sound, Amount at far right', 'each slice played twice as many extra times', ['stutter.amount=2']), ('Second sound, Amount half way left', 'each slice played about half as many extra times', ['stutter.amount=0.5'])]
f, sc, shown, ln, S = render('stutter', 's', wetA, dryA, dryB, wetB, [('closed', ['stutter.amount=0']), ('twice', ['stutter.amount=2'])], ex)
cutS = cut_of(ln['stutter'])
meta = ('%s. %s %s A stutter cannot be undone, because what the repeats covered is gone. At far left each slice is played once and the sound closes up: here it ends about %.1f seconds early, and what is left is the sound before the effect with the covered parts missing. Score for putting the stutter on the second sound: <b>%.0f</b>.%s'
        % (E(B.cap(na)), E(real_text(r)), E(stutter_text(ln['stutter'], len(wetA) / SR)), cutS, sc['onto'], E(also(shown, 'stutter', sc))))
add('s', '', panel('%d. A stutter' % N[0], meta, 'One sound: the STUTTER group',
    [row('a', 'As it came in', 'with the stutter', f['wet']), row('a', 'Amount at far left', 'each slice once, the sound closed up', f['closed']), row('a', 'Amount at far right', 'each slice played twice as many extra times', f['twice']), row('ref', 'For comparison', 'the sound before the effect, all of it', f['dry'])],
    'Two sounds: the found stutter on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'With the stutter found above', 'its own slices, at the same times', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])],
    ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(ex))), case=r['name'], scores=sc, shown=shown, cut_s=cutS)
# ---------------------------------------------------------------- a file at 44.1 kHz with a crusher and a stutter (Research/moving/mknew.py: both44)
sys.path[:0] = ['/home/claude/defect-refect/Research/crush', '/home/claude/defect-refect/Research/stutter']; import crush, stutter
cA = truth.load(SETS + 'bitcrush/bit_swh_00/', 'A'); cB = truth.load(SETS + 'bitcrush/bit_swh_00/', 'B'); six, sixB = cA['dry'], cB['dry']; x44 = resample_poly(six, 147, 160, axis=0); b44 = resample_poly(sixB, 147, 160, axis=0)
bare = dict(kind='bare', found=True, step=[2.0 / 2 ** 8] * 2, offset=[0.0, 0.0], P=3.0, c=0.0); plan = [(0.5, 0.06, 4), (1.6, 0.11, 3), (2.7, 0.045, 8), (4.0, 0.25, 3)]
ev = lambda n: [(int(t * 44100), int(L * 44100), int(t * 44100) + int(L * 44100) * k) for t, L, k in plan if int(t * 44100) + int(L * 44100) <= n]
wet44 = stutter.play(crush.through(x44, bare), ev(len(x44)), False); first = '/tmp/claude-0/v2/newfx/mc/both44.wav'; made, _ = sf.read(first, dtype='float64', always_2d=True); assert np.abs(made - np.round(wet44 * 8388608.0) / 8388608.0).max() < 1e-6
realB = resample_poly(stutter.play(crush.through(b44, bare), ev(len(b44)), False), 160, 147, axis=0)[:len(sixB)]
f, sc, shown, ln, S = render(None, 'f', wet44, None, sixB, realB, [('closed', ['stutter.amount=0'])], [], first=first, rate=44100)
nev = int(re.search(r'events=(\d+)', ln['stutter']).group(1)); assert nev in (3, 4)
missed = '' if nev == 4 else (' The slice of 45 ms, played 8 times, fell on a rest between two notes. The crusher rounds that rest to pure silence, and silence played eight times cannot be told from a longer rest, so the app reads three of the four; on the second sound that one stutter is missing, which the score counts.')
meta = ('%s, as a file at 44.1 kHz. On it: a crusher with 256 levels that holds every value for 3 samples (14.7 kHz), and after it a stutter with four slices of 45 to 250 ms, played 3 to 8 times. Both engines read such a file at its own rate, because the change to 48 kHz that the rest of the app works at takes the values off their grid and makes the repeats unequal. %s %s%s Score for putting both on the second sound: <b>%.0f</b>.%s'
        % (E(B.cap(B.sound_name(cA['meta']['A']))), E(crush_text(ln['crush'], 44100)), E(stutter_text(ln['stutter'], len(wet44) / 44100.0)), E(missed), sc['onto_all'], E(also([s for s in shown if s not in ('bitcrush', 'stutter')], None, sc))))
add('f', '', panel('%d. A crusher and a stutter in a file at 44.1 kHz' % N[0], meta, 'One sound: the BITCRUSH and STUTTER groups',
    [row('a', 'As it came in', 'crushed, then stuttered', f['wet']), row('a', 'STUTTER Amount at far left', 'each slice once, the sound closed up; the crusher stays', f['closed'])],
    'Two sounds: both on a second sound', [row('b', 'Second sound, dry', B.sound_name(cB['meta']['B']), f['b_dry']), row('b', 'With everything found above', 'the crusher first, the stutter last', f['b_all']), row('ref', 'For comparison', 'the second sound through the same crusher and stutter, made at 44.1 kHz', f['b_real'])], ''), scores=sc, shown=shown)

open(os.path.join(HERE, 'new047_panels.html'), 'w').write('\n'.join(parts)); json.dump(info, open(os.path.join(HERE, 'new047_info.json'), 'w'), indent=1, default=float)
print('panels written |', sum(os.path.getsize(os.path.join(AUD, n)) for n in os.listdir(AUD) if n.startswith('n7_')) // 1024, 'KB of audio')
