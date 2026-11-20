# The page's section for build 0.5.1 (the fixed filter, distortion and saturation): four examples rendered by the app's
# own processing (the command-line tool drrender), as MP3s in audio/q1_*.mp3, and the examples' HTML in
# four051_panels.html (build.py puts four051_head.html and these after the section of build 0.5.0).
#   python3 four051_section.py
# All four are scoreboard cases picked by fixed rules from the scoreboard's result file (see pick and named). FILTER: of
# the cases on which the group is shown, the middle one by its score for taking the filter out, and the one that goes
# onto a second sound worst. DISTORTION and SATURATION: of the cases shown under that name, the middle one by its score
# on a second sound.
import os, sys, re, json, html, subprocess, numpy as np, soundfile as sf, scipy.signal as ss
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path[:0] = ['/home/claude/defect-refect/Research/bend', '/home/claude/defect-refect/Research/filter']
import score, truth, build as B, bfeat, bengine, ffeat
AUD = os.path.join(HERE, 'audio'); SR = 48000; TMP = '/tmp/claude-0/v2/four051/'; os.makedirs(TMP, exist_ok=True); FIRST = int(os.environ.get('FIRST', '48'))
EXE = os.environ.get('EXE', '/tmp/claude-0/v2/four/build/drrender_new'); RES = '/home/claude/defect-refect/Resources'; SETS = '/tmp/claude-0/v2/bench/sets/'; BOARD = os.environ.get('BOARD', '/tmp/claude-0/v2/bench/famboard_051.jsonl')
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
BEND = ('distortion', 'saturation'); E = html.escape
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
def render(group, tag, wetA, dryA, dryB, wetB, own, extras):
    """The app on the first sound alone and with the second. group: the names its own group may have. own: (key, [fader=value ...])
    renders of the first sound alone; extras: (label, small, [fader=value ...]) the same with the second sound, the other groups off."""
    nA, nB = len(wetA), len(dryB); pa, pb = TMP + tag + '_a.wav', TMP + tag + '_b.wav'; sf.write(pa, wetA, SR, subtype='FLOAT'); sf.write(pb, dryB, SR, subtype='FLOAT')
    others = [k + '.amount=0' for k in ALL if k not in group]; blocks = [(pb, TMP + tag + '_o2.wav', others)]
    blocks += [('-', TMP + tag + '_y%d.wav' % i, fad) for i, (_, fad) in enumerate(own)] + [(pb, TMP + tag + '_x%d.wav' % i, others + fad) for i, (_, _, fad) in enumerate(extras)]
    log = run(pa, blocks); shown = re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', log, re.M)
    S = dict(wet=wetA, dry=dryA, b_dry=dryB, b_onto=rd(blocks[0][1], nB), b_real=wetB)
    for i, (key, _) in enumerate(own): S[key] = rd(TMP + tag + '_y%d.wav' % i, nA)
    for i in range(len(extras)): S['x%d' % i] = rd(TMP + tag + '_x%d.wav' % i, nB)
    sc = dict(out=score.removed(dryA, wetA, S['out'])[0], onto=score.removed(wetB, dryB, S['b_onto'])[0])
    g = 0.89 / max(np.abs(v).max() for v in S.values()); f = {k: mp3('q1_%s_%s' % (tag, k), v, g) for k, v in S.items()}
    lines = {k: (re.search(r'^%s: (.*)$' % k, log, re.M) or [None, ''])[1] for k in ('filter', 'bend')}
    print(tag, 'shown:', shown, '|', {k: v[:230] for k, v in lines.items() if 'found=0' not in v[:8]}, '| scores', {k: round(v, 1) for k, v in sc.items()}, '| peaks', {k: round(float(np.abs(v).max()), 2) for k, v in S.items()}, flush=True)
    return f, sc, [s for s, _ in shown], lines, S
def panel(title, meta, one, rows1, two, rows2, more):
    return ('<section class="panel"><h3>%s</h3><div class="meta">%s</div><div class="sub">%s</div><div class="rows">%s</div><div class="sub">%s</div><div class="rows">%s</div>%s</section>'
            % (E(title), meta, E(one), ''.join(rows1), E(two), ''.join(rows2), ('<details><summary>The other faders on this example</summary><div class="rows">%s</div></details>' % more) if more else ''))
num = lambda v: ('%.0f' % v) if abs(v - round(v)) < 0.05 else ('%.1f' % v)
sco = lambda v: ('%.0f' % (round(v) + 0.0)).replace('-', '\u2212')       # a score, with a real minus sign
hz = lambda v: ('%.0f Hz' % (round(v / 10.0) * 10.0)) if v < 995.0 else ('%s kHz' % num(round(v / 100.0) / 10.0))
def val(s, k): m = re.search(r'(?:^| )%s=([0-9.eE+-]+)' % k, s); return float(m.group(1)) if m else None

# ---------------------------------------------------------------- a filter in words: what the app read, and what the real one does
def true_curve(c):
    """The real filter's curve in the engine's bands (dB, 0 = its pass band), from the case's response to a click."""
    n = 1 << 16; P = sum(np.abs(np.fft.rfft(c['h'][i][:n, i], n)) ** 2 for i in range(2)); f = np.fft.rfftfreq(n, 1.0 / SR); T = np.empty(ffeat.NBAND)
    for b, fc in enumerate(ffeat.FC):
        half = max(fc * (2 ** (1 / 12.0) - 1), 1.5 * SR / n); w = np.maximum(0.0, 1.0 - np.abs(f - fc) / half); T[b] = 10 * np.log10((w * P).sum() / w.sum() + 1e-30)
    typ = c['meta']['params']['type']; return T - (np.median(T[:12]) if typ == 'low' else np.median(T[-12:]) if typ == 'high' else T.max())
def describe(L):
    """As describe() in Source/Core/Filter.h: the corners where the curve passes -3 dB on the way out from its highest point, and the peak above the pass band."""
    top = int(np.argmax(L)); pk = (float(L[top]), float(ffeat.FC[top])) if L[top] > 0 else (0.0, 0.0)
    def corner(step):
        b = top
        while 0 <= b + step < len(L):
            if L[b] > -3.0 and L[b + step] <= -3.0: return float(ffeat.FC[b] * 2.0 ** (step * (L[b] + 3.0) / max(L[b] - L[b + step], 1e-9) / 12.0))
            b += step
        return 0.0
    return dict(lo=corner(-1) if L[0] <= -6.0 else 0.0, hi=corner(+1) if L[-1] <= -6.0 else 0.0, peak_db=pk[0], peak_hz=pk[1])
def slope_of(L, up):
    """As slopeOf() in Source/Core/Filter.h: the skirt's fall in dB per octave between -6 dB and its far part (0: not deep enough)."""
    top = int(np.argmax(L)); step = 1 if up else -1; deepest = min(0.0, float((L[top:] if up else L[:top + 1]).min()))
    if deepest > -12.0: return 0.0
    far = max(-30.0, 0.8 * deepest)
    def passes(level):
        b = top
        while 0 <= b + step < len(L):
            if L[b] > level and L[b + step] <= level: return b + step * (L[b] - level) / max(L[b] - L[b + step], 1e-9)
            b += step
        return -1.0
    a, b = passes(-6.0), passes(far); return 0.0 if a < 0 or b < 0 or abs(b - a) < 0.25 else float((-6.0 - far) / (abs(b - a) / 12.0))
KIND = dict(low='low-pass', high='high-pass', band='band-pass')
def peak_text(db, at): return '' if db < 1.0 else ', with a peak of %s dB at %s' % (num(db), hz(at))
def read_filter(s):
    """What the filter engine read (the line drrender prints): kind, the corner that counts, the peak, the fall per octave."""
    kind = re.search(r'kind=(\w+)', s).group(1); v = {k: val(s, k) for k in ('lo_hz', 'hi_hz', 'peak_db', 'peak_hz', 'fall_up', 'fall_down')}
    return dict(kind=kind, lo=v['lo_hz'], hi=v['hi_hz'], peak_db=v['peak_db'], peak_hz=v['peak_hz'], up=v['fall_up'], down=v['fall_down'])
def filter_text(F):
    fall = lambda v, where: '' if v <= 0 else ' It falls by %.0f dB per octave %s.' % (v, where)
    if F['kind'] == 'low': return 'The app found a low-pass that is 3 dB down at %s%s.%s' % (hz(F['hi']), peak_text(F['peak_db'], F['peak_hz']), fall(F['up'], 'above that'))
    if F['kind'] == 'high': return 'The app found a high-pass that is 3 dB down at %s%s.%s' % (hz(F['lo']), peak_text(F['peak_db'], F['peak_hz']), fall(F['down'], 'below that'))
    return 'The app found a band-pass that is 3 dB down at %s and at %s%s.' % (hz(F['lo']), hz(F['hi']), peak_text(F['peak_db'], F['peak_hz']))
def real_filter(c):
    p = c['meta']['params']; D = describe(true_curve(c)); name = c['name']
    setting = 'its resonance at %.0f %%' % (100 * p['resonance']) if 'resonance' in p else 'Q %s' % num(max(0.707, p['q']) if name.startswith('fil_calf') else p['q'])
    edge = D['hi'] if p['type'] == 'low' else D['lo']; T = true_curve(c); fall = slope_of(T, True) if p['type'] == 'low' else slope_of(T, False) if p['type'] == 'high' else 0.0
    falls = '' if fall <= 0 else ', and falls by %.0f dB per octave %s that' % (fall, 'above' if p['type'] == 'low' else 'below')
    return 'The real effect: a %s set to %s, %s. Measured with a click, it is 3 dB down at %s%s%s.' % (KIND[p['type']], hz(p['cutoff']), setting, hz(edge), peak_text(D['peak_db'], D['peak_hz']), falls), dict(D, kind=p['type'], fall=fall)
def corner_off(F, D):
    """How far the corner the app read lies from the real filter's, in octaves (of the same kind: a low-pass's upper corner, a high-pass's lower one)."""
    a, b = (F['hi'], D['hi']) if D['kind'] == 'low' else (F['lo'], D['lo']); return float(np.log2(a / b)) if a > 0 and b > 0 else None

# ---------------------------------------------------------------- a bend in words
BOX = dict(dist_caps='a saturation plugin, driven hard', dist_mda='an overdrive plugin', dist_pedalboard='a plain distortion: one curve and nothing else', dist_sox='an overdrive',
           sat_calf='a saturator plugin', sat_calf_tape='a tape simulation plugin', sat_swh_valve='a valve saturation plugin', sat_tap_tube='a tube warmth plugin')
def true_top(c):
    """By how many dB the real box squeezed the loudest values, measured on the sound before and after it (as Research/bend/beval.py)."""
    x, y = c['dry'], c['wet']; cc = ss.correlate(y[:, 0] + y[:, 1], x[:, 0] + x[:, 1], 'full', 'fft'); n = len(x); k = np.arange(-4, 5) + n - 1; lag = int(k[np.argmax(np.abs(cc[k]))]) - (n - 1)
    pol = 1.0 if np.sum(x * np.roll(y, -lag, 0)) >= 0 else -1.0; T = np.clip(bfeat.answer(pol * np.roll(x, lag, 0), y), 0.0, 40.0)
    return max(bengine.tidy(T[0], bengine.SPREAD)[-1], bengine.tidy(T[1], bengine.SPREAD)[-1])
def real_bend(c): t = true_top(c); return 'The real effect: %s. Measured on the sound before and after it, it squeezes the loudest values by %s dB.' % (BOX[c['meta']['effect']], num(t)), t
def bend_text(s): return 'The app found a curve that squeezes the loudest values by %s dB.' % num(val(s, 'top_db')), val(s, 'top_db')

rows = [json.loads(l) for l in open(BOARD)]; assert len(rows) == 136
def pick(R, key, weakest=False):
    """The middle one (by a score, best first) of some cases; weakest: the last."""
    R = sorted(R, key=lambda r: -r[key]); return R[-1] if weakest else R[len(R) // 2]
def named(name): return [r for r in rows if r['fam'] in BEND and name in r['shown']]
def case(r):
    d = SETS + r['fam'] + '/' + r['name'] + '/'; cA = truth.load(d, 'A'); cB = truth.load(d, 'B'); m = cA['meta']
    return cA, cB, B.sound_name(m['A']), B.sound_name(m['B'])
TITLE = dict(filter='FILTER', distortion='DISTORTION', saturation='SATURATION', reverb='REVERB', delay='DELAY', comb='COMB FILTER', filtersweep='FILTER SWEEP', tremolo='TREMOLO', autopan='AUTO-PAN', gatedrhythm='GATED RHYTHM',
             flanger='FLANGER', chorus='CHORUS', phaser='PHASER', haas='HAAS WIDTH', width='STEREO WIDTH', ringmod='RING MOD', bitcrush='BITCRUSH', stutter='STUTTER')
def also(shown, group):
    others = [s for s in shown if s not in group]; assert all(o == 'reverb' for o in others), others     # 'not there' is known of the REVERB the app shows on some of these cases; any other group needs a look first
    return '' if not others else ' On this sound the app also shows %s, which %s not there.' % (' and '.join(TITLE[o] for o in others), 'is' if len(others) == 1 else 'are')
def same(sc, r):
    """The page's renders are the scoreboard's: the same scores."""
    assert abs(sc['out'] - r['out']) < 0.5 and abs(sc['onto'] - r['onto']) < 0.5, (sc, r['out'], r['onto'])
parts = []; info = {}; N = [FIRST]
def add(tag, html_, **kw): info[tag] = kw; parts.append(html_); N[0] += 1

# ---------------------------------------------------------------- the fixed filter: the middle case, and the one that goes onto a second sound worst
shownF = [r for r in rows if r['fam'] == 'filter' and r['found']]; r1 = pick(shownF, 'out'); r2 = pick([r for r in shownF if r['name'] != r1['name']], 'onto', True)
def filter_panel(r, tag, title, weak):
    cA, cB, na, nb = case(r); own = [('out', ['filter.amount=0']), ('twice', ['filter.amount=2']), ('up', ['filter.cutoff=1.5'])]
    ex = [('Second sound, Cutoff half way left', 'the found filter an octave lower', ['filter.cutoff=0.5']), ('Second sound, Cutoff half way right', 'the found filter an octave higher', ['filter.cutoff=1.5']),
          ('Second sound, Amount at far right', 'the found curve twice', ['filter.amount=2']), ('Second sound, Resonance at far left', 'the found filter without its peak', ['filter.resonance=0'])]
    f, sc, shown, ln, S = render(('filter',), tag, cA['wet'], cA['dry'], cB['dry'], cB['wet'], own, ex); same(sc, r); F = read_filter(ln['filter']); real, D = real_filter(cA); off = corner_off(F, D)
    if F['peak_db'] < 1.0: ex.pop(); os.remove(os.path.join(AUD, 'q1_%s_x3.mp3' % tag))          # no Resonance fader on a curve without a peak (peakShown in Source/Core/Filter.h)
    notes = [] if off is None or abs(off) < 0.15 else ['the corner is read %s of an octave too %s' % (num(abs(off)), 'high' if off > 0 else 'low')]; dp = F['peak_db'] - D['peak_db']
    if abs(dp) >= 2.0: notes.append('the peak %s dB too %s' % (num(abs(dp)), 'high' if dp > 0 else 'low') if notes else 'the peak is read %s dB too %s' % (num(abs(dp)), 'high' if dp > 0 else 'low'))
    far = ' ' + B.cap(' and '.join(notes)) + '.' if notes else ''
    meta = '%s. %s %s%s Scores: taken out <b>%s</b>, put on the second sound <b>%s</b>.%s' % (E(B.cap(na)), E(real), E(filter_text(F)), E(far), sco(sc['out']), sco(sc['onto']), E(also(shown, ('filter',))))
    name = dict(low='LOW-PASS', high='HIGH-PASS', band='BAND-PASS')[F['kind']] + ' FILTER'
    add(tag, panel('%d. %s' % (N[0], title % KIND[D['kind']]), meta, 'One sound: the %s group' % name,
        [row('a', 'As it came in', 'filtered', f['wet']), row('a', 'Amount at far left', 'the filter taken out: what it cut is lifted again', f['out']), row('a', 'Amount at far right', 'the same filter once more', f['twice']),
         row('a', 'Cutoff half way right', 'the found filter out and one an octave higher in', f['up']), row('ref', 'For comparison', 'the sound before the effect', f['dry'])],
        'Two sounds: the found filter on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'Through the filter found above', 'faders in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])],
        ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(ex))), case=r['name'], scores=sc, shown=shown, read=F, real=D, corner_off=off)
filter_panel(r1, 'f1', 'A sound through a %s filter', False)
filter_panel(r2, 'f2', 'A %s filter that is read less well', True)

# ---------------------------------------------------------------- distortion and saturation: the middle case of those shown under each name
def bend_panel(name, tag, title, back):
    r = pick(named(name), 'onto'); cA, cB, na, nb = case(r); own = [('out', ['%s.amount=0' % name]), ('twice', ['%s.amount=2' % name])]
    ex = [('Second sound, Amount half way left', 'driven 12 dB less hard into the curve', ['%s.amount=0.5' % name]), ('Second sound, Amount at far right', 'driven harder into the curve, by the %s dB that the first sound was squeezed', ['%s.amount=2' % name])]
    f, sc, shown, ln, S = render(BEND, tag, cA['wet'], cA['dry'], cB['dry'], cB['wet'], own, ex); same(sc, r); assert name in shown; found, top = bend_text(ln['bend']); real, ttop = real_bend(cA)
    ex[1] = (ex[1][0], ex[1][1] % num(top), ex[1][2])                                             # drive() in Source/Core/Bend.h: half way left is a quarter of the drive, far right as many dB more as were read at the top
    worse = ' On this sound far left comes out a little worse than leaving it alone.' if sc['out'] < 0 else ''; assert sc['out'] > -15
    meta = '%s. %s %s %s Scores: taken out <b>%s</b>, put on the second sound <b>%s</b>.%s%s' % (E(B.cap(na)), E(real), E(found), E(back), sco(sc['out']), sco(sc['onto']), E(worse), E(also(shown, BEND)))
    add(tag, panel('%d. %s' % (N[0], title), meta, 'One sound: the %s group' % name.upper(),
        [row('a', 'As it came in', 'through the effect', f['wet']), row('a', 'Amount at far left', 'the bend walked back; the peak stays where it was, so the sound is quieter', f['out']), row('a', 'Amount at far right', 'pushed as far again into the same curve', f['twice']),
         row('ref', 'For comparison', 'the sound before the effect', f['dry'])],
        'Two sounds: the found curve on a second sound', [row('b', 'Second sound, dry', nb, f['b_dry']), row('b', 'Through the curve found above', 'fader in the middle: its loudest values as far into the curve as the first sound\'s were', f['b_onto']),
                                                           row('ref', 'For comparison', 'the second sound through the real effect', f['b_real'])],
        ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(ex))), case=r['name'], scores=sc, shown=shown, top=top, true_top=ttop)
bend_panel('distortion', 'd', 'A distorted sound', 'Far left spreads the squeezed levels out again, but tops that were pressed flat stay flat: nothing in the sound says how high each of them stood.')
bend_panel('saturation', 's', 'A saturated sound', 'Far left spreads the squeezed levels out again, most of the way: a little of the bend is left in, to be on the safe side.')

open(os.path.join(HERE, 'four051_panels.html'), 'w').write('\n'.join(parts)); json.dump(info, open(os.path.join(HERE, 'four051_info.json'), 'w'), indent=1, default=float)
print('panels written |', sum(os.path.getsize(os.path.join(AUD, n)) for n in os.listdir(AUD) if n.startswith('q1_')) // 1024, 'KB of audio')
