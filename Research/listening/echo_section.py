# The page's Echo section (build 0.4.2): two examples rendered by the app's own processing (the command-line tool drrender), as MP3s
# in audio/echo_*.mp3, and the section's HTML in echo.html (build.py puts it after the tone section).
#   python3 echo_section.py
import os, sys, re, json, html, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); import score, truth
HERE = os.path.dirname(os.path.abspath(__file__)); AUD = os.path.join(HERE, 'audio'); SR = 48000; TMP = '/dev/shm/ep/'
EXE = '/tmp/claude-0/drbuild/drrender'; RES = '/home/claude/defect-refect/Resources'; S = '/tmp/claude-0/v2/bench/sets/delay/'
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
OTHERS = [k + '.amount=0' for k in ALL if k != 'delay']
def mp3(name, x, gain):
    wav = os.path.join(AUD, name + '.wav'); y = np.clip(x * gain, -1.0, 1.0); n = len(y); f = min(480, n // 4); y[:f] *= np.linspace(0, 1, f)[:, None]; y[-f:] *= np.linspace(1, 0, f)[:, None]
    sf.write(wav, y, SR, subtype='PCM_24'); subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame', '-b:a', '256k', os.path.join(AUD, name + '.mp3')], check=True); os.remove(wav)
    return 'audio/%s.mp3' % name
def row(cls, label, small, src): return '<div class="row %s"><div class="lab">%s<small>%s</small></div><audio controls preload="none" src="%s"></audio></div>' % (cls, html.escape(label), html.escape(small), src)
def run(first, blocks):
    """One analysis of the first sound, several renders: blocks = (second or '-', out, [fader=value ..]). Returns what the app printed."""
    args = []
    for i, (second, out, fad) in enumerate(blocks): args += (['--'] if i else []) + [second, out] + list(fad)
    r = subprocess.run([EXE, RES, first] + args, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
    return r.stdout
def rd(p, n): y, _ = sf.read(p, dtype='float64', always_2d=True); return y[:n]
def example(case, tag, extras):
    cA = truth.load(S + case + '/', 'A'); cB = truth.load(S + case + '/', 'B'); nA, nB = len(cA['wet']), len(cB['dry'])
    pa, pb = TMP + '_pg_a.wav', TMP + '_pg_b.wav'; sf.write(pa, cA['wet'], SR, subtype='FLOAT'); sf.write(pb, cB['dry'], SR, subtype='FLOAT')
    blocks = [('-', TMP + '_pg_o0.wav', ['delay.amount=0']), ('-', TMP + '_pg_o1.wav', [k + '.amount=0' for k in ALL]), (pb, TMP + '_pg_o2.wav', OTHERS), (pb, TMP + '_pg_o3.wav', [])]
    for i, (_, _, fad) in enumerate(extras): blocks.append((pb, TMP + '_pg_x%d.wav' % i, OTHERS + fad))
    log = run(pa, blocks); shown = re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', log, re.M); m = re.search(r'echo: sure=([0-9.]+)(.*)', log)
    x = dict(wet=cA['wet'], out=rd(blocks[0][1], nA), out_all=rd(blocks[1][1], nA), dry=cA['dry'], b_dry=cB['dry'], b_onto=rd(blocks[2][1], nB), b_all=rd(blocks[3][1], nB), b_real=cB['wet'])
    for i in range(len(extras)): x['x%d' % i] = rd(TMP + '_pg_x%d.wav' % i, nB)
    g = 0.89 / max(np.abs(v).max() for v in x.values()); f = {k: mp3('echo_%s_%s' % (tag, k), v, g) for k, v in x.items()}
    sc = dict(out=score.removed(cA['dry'], cA['wet'], x['out'])[0], out_all=score.removed(cA['dry'], cA['wet'], x['out_all'])[0], onto=score.removed(cB['wet'], cB['dry'], x['b_onto'])[0], onto_all=score.removed(cB['wet'], cB['dry'], x['b_all'])[0])
    lag = dict(re.findall(r'(left|right)_ms=([0-9.]+)', m.group(2))) if m else {}
    print(case, 'shown:', shown, '| echo', m.group(0) if m else None, '| scores', {k: round(v, 1) for k, v in sc.items()}, flush=True)
    return f, sc, [s for s, _ in shown], lag
E = html.escape; parts = []; info = {}
EX = [('dly_pedalboard_01', 'p', '10. Computer speech with an echo that repeats', 'The real effect: an echo every 408 ms, each repeat a quarter of the one before.', 'chords on a synthesizer',
       [('Second sound, Time at far left', 'the same echoes at half the echo time', ['delay.time=0']), ('Second sound, Feedback at far right', 'the echoes go on for longer', ['delay.feedback=2'])]),
      ('dly_tap_00', 't', '11. Slap bass notes with a different echo on each side', 'The real effect: an echo every 432 ms on the left and every 324 ms on the right, each fed back a little.', 'a synthetic engine sound',
       [('Second sound, Amount at far right', 'the echoes twice as loud', ['delay.amount=2'])])]
for case, tag, title, real, second, extras in EX:
    f, sc, shown, lag = example(case, tag, extras); info[case] = dict(scores=sc, shown=shown, lag=lag)
    found = 'The app found an echo every %s.' % (('%.0f ms' % float(lag['left'])) if lag.get('left') == lag.get('right') else ' and '.join('%.0f ms on the %s' % (float(v), k) for k, v in lag.items()))
    others = [s for s in shown if s != 'delay']
    note = ('' if not others else ' On this sound the app also shows %s, which %s not there; the two rows marked "everything shown" are what comes out when those are moved as well.' % (', '.join(o.upper() for o in others), 'is' if len(others) == 1 else 'are'))
    rows1 = [row('a', 'As it came in', 'with the echoes on it', f['wet']), row('a', 'Amount at far left', 'the echoes taken out', f['out'])]
    rows2 = [row('b', 'Second sound, dry', second, f['b_dry']), row('b', 'With the echoes found above', 'fader in the middle', f['b_onto'])]
    if others: rows1.append(row('a', 'Everything shown at far left', 'score %.0f' % sc['out_all'], f['out_all'])); rows2.append(row('b', 'Everything shown, faders in the middle', 'score %.0f' % sc['onto_all'], f['b_all']))
    rows1.append(row('ref', 'For comparison', 'the sound before the echoes were added', f['dry'])); rows2.append(row('ref', 'For comparison', 'the second sound through the real effect', f['b_real']))
    more = ''.join(row('b', lab, small, f['x%d' % i]) for i, (lab, small, _) in enumerate(extras))
    parts.append('<section class="panel"><h3>%s</h3><div class="meta">%s %s Scores: echoes taken out <b>%.0f</b>, put on the second sound <b>%.0f</b>.%s</div>'
                 '<div class="sub">One sound: the DELAY group</div><div class="rows">%s</div><div class="sub">Two sounds: the found echoes on a second sound</div><div class="rows">%s</div>'
                 '<details><summary>The other faders on this example</summary><div class="rows">%s</div></details></section>' % (E(title), E(real), E(found), sc['out'], sc['onto'], E(note), ''.join(rows1), ''.join(rows2), more))
head = open(os.path.join(HERE, 'echo_head.html')).read() if os.path.exists(os.path.join(HERE, 'echo_head.html')) else '<section id="echo"><h2>Echoes: new in build 0.4.2</h2></section>'
open(os.path.join(HERE, 'echo.html'), 'w').write(head + '\n' + '\n'.join(parts)); json.dump(info, open(os.path.join(HERE, 'echo_info.json'), 'w'), indent=1)
print('echo section written |', sum(os.path.getsize(os.path.join(AUD, n)) for n in os.listdir(AUD) if n.startswith('echo_')) // 1024, 'KB of echo audio')
