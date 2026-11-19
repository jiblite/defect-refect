# Builds the listening page for the V2 reverb engine: picks the examples by a fixed rule from the scored test cases, renders
# every example with the engine as the app will run it, writes MP3s to audio/ and the page to index.html (the page loads its
# own audio files by relative path).
#   MODEL=<name> python3 build.py --render      render the audio again (otherwise the stored rendered.json is used)
import sys, os, json, subprocess, html
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); AUD = os.path.join(HERE, 'audio'); os.makedirs(AUD, exist_ok=True)
MODEL = os.environ.get('MODEL', 'unet_c'); SETS = '/tmp/claude-0/v2/bench/sets/reverb/'; SR = 48000
RANKS = [int(v) for v in os.environ.get('RANKS', '3,10,17,24,31,38').split(',')]         # places in the ranking by "reverb out", best first

# ---------------------------------------------------------------- plain names for the test material
INSTR = dict(altosax='alto sax', cleangtr='clean guitar', ebass='electric bass', epiano='electric piano', nylongtr='nylon guitar', pizz='plucked strings', sawlead='saw lead',
             squarelead='square lead', steeldrum='steel drum', steelgtr='steel guitar', vibes='vibraphone', drawbar='drawbar organ', polysynth='poly synth', warmpad='warm pad')
def sound_name(a):
    p = a.split('_')
    if a.startswith('real_'): return 'one of your recordings'
    if a.startswith('tts_'): return 'computer speech'
    if a.startswith('syn_'): return dict(burst='noise bursts', engine='a synthetic engine tone', fm='a synthetic bell tone', noise='noise with gaps', sweep='a tone sweep')[p[1]]
    if a.startswith('fs_mel_'): n = INSTR.get(p[2], p[2]); return '%s %s melody' % ('an' if n[0] in 'aeiou' else 'a', n)
    if a.startswith('fs_chd_'): return '%s chords' % INSTR.get(p[2], p[2])
    if a.startswith('fs_stab_'): return '%s hits' % INSTR.get(p[2], p[2])
    if a.startswith('fs_drm_'): return 'a drum kit pattern'
    if a.startswith('fs_band_'): return 'a full band (drums, bass, chords, melody)'
    if a.startswith('sp_seq_') or a.startswith('lm_seq_'): return dict(bd='a kick drum pattern', sn='a snare pattern', perc='a percussion pattern', tabla='a tabla pattern', elec='an electronic drum pattern').get(p[2], 'a drum machine pattern')
    if a.startswith('lm_ins_'): return 'single notes (%s)' % ' '.join(p[2:]).rstrip('0123456789').replace('e organ', 'electric organ').replace('e piano accord', 'electric piano chord').replace('bassslap', 'slap bass')
    if 'bass' in a or 'tb303' in a or 'briff' in a: return 'a bass loop'
    if 'guit' in a: return 'a guitar sample'
    if 'phone' in a: return 'a telephone sample'
    if 'drone' in a: return 'a drone'
    if any(k in a for k in ('break', 'beat', 'amen', 'jungle', 'house', 'perc', 'tabla')): return 'a drum loop'
    return 'a sample loop'
EFFECT = dict(rev_calf='a room and hall reverb plugin', rev_caps_plate='a plate reverb plugin', rev_dragonfly_hall='a hall reverb plugin', rev_dragonfly_plate='a plate reverb plugin',
              rev_dragonfly_room='a room reverb plugin', rev_gverb='an older algorithmic reverb', rev_mda_ambience='a short ambience plugin', rev_pedalboard='a classic studio-style reverb',
              rev_room_large='a simulated large room', rev_room_medium='a simulated medium room', rev_room_small='a simulated small room', rev_sox='a classic free reverb',
              rev_swh_plate='a plate reverb plugin', rev_tap='an older reverb plugin', rev_zita='a hall reverb plugin')
def cap(s): return s[0].upper() + s[1:]
def wetness(drr): return 'light reverb' if drr >= 8 else 'medium reverb' if drr >= 0 else 'heavy reverb (the reverb is louder than the sound itself)'

def pick():
    """The examples: fixed places in the ranking of the scored cases by 'reverb out' (best first)."""
    rows = json.load(open('/tmp/claude-0/v2/train/final_eval_%s.json' % MODEL)); rows.sort(key=lambda r: -r['out']['reverb out']); out = []
    for k in RANKS:
        r = rows[min(k, len(rows)) - 1]; m = json.load(open(SETS + r['name'] + '/meta.json'))
        out.append(dict(case=r['name'], rank=k, of=len(rows), title='%s through %s' % (cap(sound_name(m['A'])), EFFECT[m['effect']]), second=sound_name(m['B']), wet=wetness(r['drr']), drr=r['drr'],
                        s_out=r['out']['reverb out'], s_bold=r['out']['bolder'], s_more=r['out']['doubled'], s_onto=r['out']['room onto second sound'], rev_cut=r.get('rev_cut'), dir_cut=r.get('dir_cut')))
    return out

def mp3(name, x, gain):
    wav = os.path.join(AUD, name + '.wav'); y = np.clip(x * gain, -1.0, 1.0); n = len(y); f = min(480, n // 4); y[:f] *= np.linspace(0, 1, f)[:, None]; y[-f:] *= np.linspace(1, 0, f)[:, None]
    sf.write(wav, y, SR, subtype='PCM_24'); subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame', '-b:a', '256k', os.path.join(AUD, name + '.mp3')], check=True); os.remove(wav)
    return 'audio/%s.mp3' % name

def render_all(EX):
    import truth, engine, common as cm, rc_look, param_ceiling as pc
    out = []
    for i, ex in enumerate(EX):
        d = SETS + ex['case'] + '/'; cA = truth.load(d, 'A'); direct, rev = truth.parts(cA); an = engine.analyse(cA['wet'], MODEL); cB = truth.load(d, 'B'); tag = 'ex%02d' % (i + 1)
        a0 = max(0, int((cA['meta'].get('pre', 0.0) - 0.15) * SR)); cut = lambda v: v[a0:]           # the test files start with 1.5 s of silence: not needed for listening
        one = dict(wet=cA['wet'], out=engine.fader(cA['wet'], an, 0.5), bold=engine.fader(cA['wet'], an, 0.0), more=engine.fader(cA['wet'], an, 2.0), ref=direct)       # the Amount fader as the app has it
        if an['found']: one.update(shorter=engine.fader(cA['wet'], an, 1.0, length=0.5), longer=engine.fader(cA['wet'], an, 1.0, length=1.6))
        two = dict(dry=cB['dry'], onto=engine.onto(cB['dry'], an, 1.0), real=cB['wet'], short=engine.onto(cB['dry'], an, 1.0, length=0.5), long=engine.onto(cB['dry'], an, 1.0, length=2.0)) if an['found'] else None
        g1 = 0.89 / max(np.abs(v).max() for v in one.values()); files = {k: mp3('%s_a_%s' % (tag, k), cut(v), g1) for k, v in one.items()}
        if two: g2 = 0.89 / max(np.abs(v).max() for v in two.values()); files.update({'b_' + k: mp3('%s_b_%s' % (tag, k), cut(v), g2) for k, v in two.items()})
        mid = (cm.FC41 > 450) & (cm.FC41 < 7000); num = dict(found=bool(an['found']))
        ft = cm.features(cA['wet'], direct); kt = 10 * np.log10((10 ** (ft['LR'] / 10)).sum(0) / np.maximum((10 ** (ft['LS'] / 10)).sum(0), 1e-30) + 1e-9); Tc = np.array([pc.t60_of(x) for x in rc_look.click_W(cA)]); ok = mid & np.isfinite(Tc)
        num.update(true_t60=float(np.median(Tc[ok])) if ok.any() else None, true_kdb=float(np.median(kt[mid])))
        if an['found']: num.update(t60=float(np.median(an['T60'][mid])), t60_low=float(np.median(an['T60'][cm.FC41 < 350])), t60_high=float(np.median(an['T60'][cm.FC41 > 5000])), kdb=float(np.median(an['kdb'][mid])), width=float(np.median(an['width'][mid])))
        out.append(dict(ex=ex, files=files, num=num)); print(tag, ex['case'], num, flush=True)
    json.dump(out, open(os.path.join(HERE, 'rendered.json'), 'w'), indent=1); return out

CSS = '''
:root {
  /* layout: one reading column; each example is a panel with two stacks of labelled players. teal = first sound, orange = second sound, as in the app */
  --bg: #f3f5f4; --panel: #ffffff; --ink: #17201f; --muted: #5d6b69; --line: #d5dcda; --teal: #0f8f86; --orange: #c7661a; --ref: #6b7775;
  --display: "Barlow Semi Condensed", "Arial Narrow", system-ui, sans-serif; --body: "Barlow", system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #101514; --panel: #18201f; --ink: #e6eeec; --muted: #9aa9a6; --line: #2b3735; --teal: #3fc4b8; --orange: #f09a4a; --ref: #8d9a97; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #101514; --panel: #18201f; --ink: #e6eeec; --muted: #9aa9a6; --line: #2b3735; --teal: #3fc4b8; --orange: #f09a4a; --ref: #8d9a97; color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 17px; line-height: 1.5; }
main { max-width: 760px; margin: 0 auto; padding: 28px 18px 64px; display: flex; flex-direction: column; gap: 28px; }
h1 { font-family: var(--display); font-weight: 600; font-size: 34px; line-height: 1.1; margin: 0 0 10px; text-wrap: balance; letter-spacing: 0.01em; }
h2 { font-family: var(--display); font-weight: 600; font-size: 22px; margin: 0 0 8px; text-wrap: balance; text-transform: uppercase; letter-spacing: 0.06em; }
h3 { font-family: var(--display); font-weight: 600; font-size: 21px; line-height: 1.2; margin: 0; text-wrap: balance; }
p { margin: 0 0 10px; max-width: 65ch; }
p:last-child { margin-bottom: 0; }
.lead { color: var(--muted); }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.meta { color: var(--muted); font-size: 15px; font-variant-numeric: tabular-nums; }
.meta b { color: var(--ink); font-weight: 500; }
.rows { display: flex; flex-direction: column; gap: 10px; }
.row { display: grid; grid-template-columns: 220px 1fr; gap: 10px; align-items: center; }
.row .lab { font-weight: 500; border-left: 4px solid var(--line); padding-left: 10px; }
.row .lab small { display: block; color: var(--muted); font-weight: 400; font-size: 14px; line-height: 1.3; }
.row.a .lab { border-color: var(--teal); } .row.b .lab { border-color: var(--orange); } .row.ref .lab { border-color: var(--ref); border-left-style: dashed; }
audio { width: 100%; height: 40px; }
details summary { cursor: pointer; font-family: var(--display); text-transform: uppercase; letter-spacing: 0.08em; font-size: 14px; color: var(--muted); padding: 4px 0; }
details[open] summary { margin-bottom: 10px; }
.sub { font-family: var(--display); text-transform: uppercase; letter-spacing: 0.08em; font-size: 14px; color: var(--muted); margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 16px; }
th, td { text-align: left; padding: 7px 10px 7px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-family: var(--display); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; font-size: 14px; color: var(--muted); }
td.n, th.n { text-align: right; padding-right: 0; padding-left: 10px; }
.tablewrap { overflow-x: auto; }
@media (max-width: 560px) { .row { grid-template-columns: 1fr; gap: 4px; } body { font-size: 16px; } h1 { font-size: 28px; } }

/* a pitch curve as the app read it (build 0.5.0): one line; the axis texts are HTML so that they keep their size on a phone. --curve: the first sound's teal, stepped so that it holds as a thin line on both surfaces */
:root { --curve: #0a9087; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --curve: #2aa89d; } }
:root[data-theme="dark"] { --curve: #2aa89d; }
figure.curve { margin: 0; }
figure.curve figcaption { color: var(--muted); font-size: 15px; }
.plot { position: relative; height: 150px; margin: 14px 6px 30px 48px; }
.plot svg { display: block; width: 100%; height: 100%; overflow: visible; outline-offset: 6px; touch-action: pan-y; cursor: crosshair; }
.plot .none { fill: var(--muted); opacity: 0.14; }
.plot .grid { stroke: var(--line); stroke-width: 1; }
.plot .line { fill: none; stroke: var(--curve); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.plot .yt, .plot .xt, .plot .nr { position: absolute; color: var(--muted); font-size: 13px; line-height: 1; font-variant-numeric: tabular-nums; white-space: nowrap; pointer-events: none; }
.plot .yt { right: 100%; padding-right: 8px; transform: translateY(-50%); }
.plot .xt { top: 100%; padding-top: 8px; transform: translateX(-50%); }
.plot .nr { top: 7px; text-align: center; }
.plot .hair { position: absolute; top: 0; bottom: 0; width: 1px; margin-left: -0.5px; background: var(--muted); display: none; pointer-events: none; }
.plot .dot { position: absolute; width: 8px; height: 8px; margin: -4px 0 0 -4px; border-radius: 50%; background: var(--curve); box-shadow: 0 0 0 2px var(--panel); display: none; pointer-events: none; }
.plot .tip { position: absolute; bottom: 100%; margin-bottom: 6px; z-index: 2; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 3px 8px; font-size: 14px; line-height: 1.3; white-space: nowrap; font-variant-numeric: tabular-nums; display: none; pointer-events: none; }
.plot .tip b { font-weight: 500; }
@media (max-width: 560px) { .plot { margin-left: 44px; } .plot .nr { display: none; } }
'''
JS = '''
document.addEventListener('play', function (e) { document.querySelectorAll('audio').forEach(function (a) { if (a !== e.target) a.pause(); }); }, true);
/* the pitch curve pictures: a hairline and the value under the pointer, or under the arrow keys when the picture has the focus */
document.querySelectorAll('.plot').forEach(function (plot) {
  var svg = plot.querySelector('svg'), tip = plot.querySelector('.tip'), hair = plot.querySelector('.hair'), dot = plot.querySelector('.dot');
  var c = svg.dataset.c.split(',').map(Number), h = svg.dataset.h, n = c.length, lo = +svg.dataset.lo, hi = +svg.dataset.hi, at = -1;
  function show(i) {
    at = Math.max(0, Math.min(n - 1, i)); var x = 100 * at / (n - 1), y = 100 * (hi - c[at]) / (hi - lo);
    hair.style.left = x + '%'; hair.style.display = 'block'; dot.style.left = x + '%'; dot.style.top = y + '%'; dot.style.display = 'block';
    tip.textContent = ''; var b = document.createElement('b'); b.textContent = (c[at] > 0 ? '+' : '') + c[at] + ' cents'; tip.appendChild(b);
    tip.appendChild(document.createTextNode(' at ' + (at / 100).toFixed(2) + ' s' + (h.charAt(at) === '1' ? '' : ', not read')));
    tip.style.display = 'block'; var w = plot.clientWidth, tw = tip.offsetWidth; tip.style.left = Math.max(-44, Math.min(w - tw, w * x / 100 - tw / 2)) + 'px';
  }
  function hide() { at = -1; hair.style.display = dot.style.display = tip.style.display = 'none'; }
  function point(e) { var r = svg.getBoundingClientRect(); show(Math.round((e.clientX - r.left) / r.width * (n - 1))); }
  svg.addEventListener('pointermove', point); svg.addEventListener('pointerdown', point); svg.addEventListener('pointerleave', hide); svg.addEventListener('blur', hide);
  svg.addEventListener('focus', function () { if (at < 0) show(Math.round((n - 1) / 2)); });
  svg.addEventListener('keydown', function (e) { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); show((at < 0 ? Math.round((n - 1) / 2) : at) + (e.key === 'ArrowLeft' ? -10 : 10)); } });
});
'''
def row(cls, label, small, src):
    return '<div class="row %s"><div class="lab">%s<small>%s</small></div><audio controls preload="none" src="%s"></audio></div>' % (cls, html.escape(label), html.escape(small), src)
def secs(v): return ('%.1f s' % v) if v >= 0.95 else ('%.2f s' % v)
def page(items, intro, table_html, outro, extra=''):
    parts = []
    for i, it in enumerate(items):
        ex, f, n = it['ex'], it['files'], it['num']; E = html.escape
        truth_ = 'The real reverb: decay time %s, room level %+.0f dB next to the sound.' % (secs(n['true_t60']), n['true_kdb']) if n.get('true_t60') else ''
        found = ('Found: decay time <b>%s</b> (%s low, %s high), room level <b>%+.0f dB</b>, width %d %%. %s' % (secs(n['t60']), secs(n['t60_low']), secs(n['t60_high']), n['kdb'], round(100 * n['width']), E(truth_))) if n.get('found') else 'No room found on this sound. ' + E(truth_)
        score = "Scores: the engine's own answer <b>%.0f</b>, the bolder end %.0f, doubled %.0f%s. Place %d of %d." % (ex['s_out'], ex['s_bold'], ex['s_more'], (', found room on the second sound <b>%.0f</b>' % ex['s_onto']) if n.get('found') else '', ex['rank'], ex['of'])
        rows = [row('a', 'As it came in', 'with the reverb on it', f['wet']), row('a', "The engine's own answer", 'Amount at far left since build 0.4.6 (half way left before)', f['out']), row('a', 'The bolder end', 'far left in builds 0.4.1 to 0.4.5: digs deeper, takes more risk. Gone since build 0.4.6', f['bold']), row('a', 'Amount at far right', 'the reverb doubled', f['more']), row('ref', 'For comparison', 'the sound before the reverb was added', f['ref'])]
        rows2 = [row('b', 'Second sound, dry', ex['second'], f['b_dry']), row('b', 'With the room found above', 'fader in the middle', f['b_onto']), row('ref', 'For comparison', 'the second sound through the real reverb', f['b_real'])] if 'b_onto' in f else []
        more = ([row('a', 'First sound, Length half way left', 'its own reverb made half as long', f['shorter']), row('a', 'First sound, Length towards the right', 'its own reverb made 1.6 times as long', f['longer'])] if 'shorter' in f else []) + ([row('b', 'Second sound, Length half way left', 'the found room, half as long', f['b_short']), row('b', 'Second sound, Length at far right', 'the found room, twice as long', f['b_long'])] if 'b_short' in f else [])
        parts.append('<section class="panel"><h3>%d. %s</h3><div class="meta">%s. %s</div><div class="meta">%s</div><div class="sub">One sound: the Amount fader</div><div class="rows">%s</div>%s%s</section>' % (
            i + 1, E(ex['title']), E(cap(ex['wet'])), score, found, ''.join(rows), ('<div class="sub">Two sounds: the found room on a second sound</div><div class="rows">%s</div>' % ''.join(rows2)) if rows2 else '',
            ('<details><summary>The Length fader on this example</summary><div class="rows">%s</div></details>' % ''.join(more)) if more else ''))
    return '''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Defect Refect listening</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500&family=Barlow+Semi+Condensed:wght@600&display=swap" rel="stylesheet">
<style>%s</style></head><body><main>
<header><h1>The new engines, to listen to: reverb, tone, echo, level movement and flanger</h1>%s</header>
%s
%s
%s
<section>%s</section>
</main><script>%s</script></body></html>''' % (CSS, intro, table_html, '\n'.join(parts), extra, outro, JS)

if __name__ == '__main__':
    exf = os.path.join(HERE, 'examples.json')
    if '--render' in sys.argv or not os.path.exists(os.path.join(HERE, 'rendered.json')):
        EX = pick(); json.dump(EX, open(exf, 'w'), indent=1); items = render_all(EX)
    else: items = json.load(open(os.path.join(HERE, 'rendered.json')))
    rd = lambda name: open(os.path.join(HERE, name)).read() if os.path.exists(os.path.join(HERE, name)) else ''
    open(os.path.join(HERE, 'index.html'), 'w').write(page(items, rd('intro.html'), rd('table.html'), rd('outro.html'), rd('tone.html') + '\n' + rd('echo.html') + '\n' + rd('level.html') + '\n' + rd('new044.html') + '\n' + rd('cut045_head.html') + '\n' + rd('cut045_panels.html') + '\n' + rd('fader046_head.html') + '\n' + rd('fader046_panels.html') + '\n' + rd('new047_head.html') + '\n' + rd('new047_panels.html') + '\n' + rd('move050_head.html') + '\n' + rd('move050_panels.html')))
    print('page written,', len(items), 'examples,', sum(os.path.getsize(os.path.join(AUD, f)) for f in os.listdir(AUD)) // 1024, 'KB of audio')
