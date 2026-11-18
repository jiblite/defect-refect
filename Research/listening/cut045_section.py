# The page's section for build 0.4.5 (the room model trained again for files that stop while the sound plays and for
# excerpts): examples rendered by the app's own processing (the command-line tool drrender) as build 0.4.4 has it and as
# build 0.4.5 has it, as MP3s in audio/ct*_*.mp3, and the examples' HTML in cut045_panels.html (build.py puts cut045_head.html and these after the 0.4.4 section).
#   python3 cut045_section.py
# The examples are picked by a fixed rule from the 40 test reverbs of train/cut_app.py (see PICK below).
import os, sys, re, json, html, glob, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import score, truth, build as B
AUD = os.path.join(HERE, 'audio'); SR = 48000; TMP = '/tmp/claude-0/v2/cutapp/'; os.makedirs(TMP, exist_ok=True)
OLD = ('/tmp/claude-0/lvx/drrender_044', '/tmp/claude-0/v2/res_044'); NEW = ('/tmp/claude-0/v2/bench/drrender_045b', '/home/claude/defect-refect/Resources'); SETS = '/tmp/claude-0/v2/bench/sets/reverb/'
E = html.escape; CUT = 6 * SR; E0, E1 = SR, int(5.5 * SR); FIRST = int(os.environ.get('FIRST', '24'))
def rows_of(tag): return {r['name']: r for f in sorted(glob.glob('/tmp/claude-0/v2/train/cut_app_%s_*.jsonl' % tag)) for r in map(json.loads, open(f))}
def sc(direct, wet, y): s = score.Scorer(direct); return 100.0 * (1.0 - s(y) / max(s(wet), 1e-9))
def app(exe_res, path, n, log=None):
    """The app on one file: shown, the sureness that counted, room found, Amount half way left, Amount far left."""
    p1, p2 = TMP + 'half.wav', TMP + 'left.wav'; q = subprocess.run([exe_res[0], exe_res[1], path, '-', p1, 'reverb.amount=0.5', '--', '-', p2, 'reverb.amount=0'], capture_output=True, text=True)
    if q.returncode: raise RuntimeError(q.stdout[-300:] + q.stderr[-300:])
    if log: open(log, 'w').write(q.stdout)
    shown = re.search(r'^reverb p=([0-9.eE+-]+) \(room engine\)', q.stdout, re.M); rm = re.search(r'^room: found=(\d) .*? sureness=([0-9.eE+-]+)', q.stdout, re.M)
    counts = re.findall(r'^room with[^\n]*? counts=([0-9.eE+-]+)', q.stdout, re.M); found = bool(int(rm.group(1))); first = float(rm.group(2))      # a second look of the build, when it took one, has the last word
    rd = lambda p: sf.read(p, dtype='float64', always_2d=True)[0][:n]
    return dict(shown=bool(shown), p=float(shown.group(1)) if shown else 0.0, found=found, first=first, sure=float(counts[-1]) if counts else first, again=bool(counts), half=rd(p1), left=rd(p2))
def pick():
    """Fixed rule, on the 40 test reverbs of cut_app.py. File cut off at 6 s: (1) of the cases on which both builds show the
    REVERB group, ranked by how much the score of Amount half way left changed from 0.4.4 to 0.4.5, place 3 (a strong change)
    and the middle one (a typical change); (2) of the cases on which only 0.4.5 shows it, ranked by its score, the middle one;
    (3) the cases on which only 0.4.4 showed it: the middle one. Excerpt: of the cases both builds show, the middle one."""
    a, b = rows_of('044'), rows_of('045'); out = []
    both = sorted((n for n in b if a[n]['cut']['shown'] and b[n]['cut']['shown']), key=lambda n: -(b[n]['cut']['out'] - a[n]['cut']['out']))
    out += [(both[2], 'cut', 'both builds show it: place 3 of %d by the change' % len(both)), (both[len(both) // 2], 'cut', 'both builds show it: the middle of %d by the change' % len(both))]
    new = sorted((n for n in b if b[n]['cut']['shown'] and not a[n]['cut']['shown']), key=lambda n: -b[n]['cut']['out']); out.append((new[len(new) // 2], 'cut', 'only 0.4.5 shows it: the middle of %d by its score' % len(new)))
    gone = sorted((n for n in b if a[n]['cut']['shown'] and not b[n]['cut']['shown']), key=lambda n: a[n]['cut']['out'])
    if gone: out.append((gone[len(gone) // 2], 'cut', 'only 0.4.4 showed it: the middle of %d' % len(gone)))
    both = sorted((n for n in b if a[n]['excerpt']['shown'] and b[n]['excerpt']['shown']), key=lambda n: -(b[n]['excerpt']['out'] - a[n]['excerpt']['out']))
    out.append((both[len(both) // 2], 'excerpt', 'both builds show it: the middle of %d by the change' % len(both)))
    return out
if __name__ == '__main__':
    EX = pick(); parts = []; info = []; a44, a45 = rows_of('044'), rows_of('045')
    for i, (case, form, how) in enumerate(EX):
        c = truth.load(SETS + case + '/', 'A'); direct, rev = truth.parts(c); m = c['meta']; lo, hi = (0, CUT) if form == 'cut' else (E0, E1); w, dd = c['wet'][lo:hi], direct[lo:hi]; tag = 'ct%d' % (i + 1)
        p = TMP + tag + '.wav'; sf.write(p, w, SR, subtype='FLOAT'); o, n = app(OLD, p, len(w), TMP + tag + '_044.log'), app(NEW, p, len(w), TMP + tag + '_045.log')
        s = {k: (sc(dd, w, o[k]) if o['shown'] else 0.0) for k in ('half', 'left')}; s2 = {k: (sc(dd, w, n[k]) if n['shown'] else 0.0) for k in ('half', 'left')}
        a0 = max(0, int((m.get('pre', 0.0) - 0.15) * SR)) if form == 'cut' else 0        # the test files start with silence: not needed for listening
        x = dict(wet=w, ref=dd)
        if o['shown']: x.update(o_half=o['half'], o_left=o['left'])
        if n['shown']: x.update(n_half=n['half'], n_left=n['left'])
        g = 0.89 / max(np.abs(v).max() for v in x.values()); f = {k: B.mp3('%s_%s' % (tag, k), v[a0:], g) for k, v in x.items()}
        drr = float(10 * np.log10(max((dd ** 2).sum(), 1e-20) / max(((w - dd) ** 2).sum(), 1e-20)))
        title = '%d. %s through %s, %s' % (FIRST + i, B.cap(B.sound_name(m['A'])), B.EFFECT[m['effect']], 'cut off while it plays' if form == 'cut' else 'an excerpt cut at both ends')
        def said(v, lab):
            if v['shown']: return 'Build %s shows REVERB at %.0f %% sure' % (lab, 100 * v['p'])
            why = 'it finds no room' if not v['found'] else ('it is %.0f %% sure%s, and it takes 65 %%' % (100 * v['sure'], (' after its second look, %.0f %% on the first' % (100 * v['first'])) if v['again'] and v['sure'] < v['first'] - 0.005 else ''))
            return 'Build %s shows no REVERB group here (%s), so the file stays as it is' % (lab, why)
        meta = '%s. %s; %s.' % (E(B.cap(B.wetness(drr))), E(said(o, '0.4.4')), E(said(n, '0.4.5').replace('Build', 'build', 1)))
        if o['shown'] or n['shown']: meta += ' Scores (100 = the sound before the reverb, 0 = no better than untouched): half way left <b>%.0f</b> before and <b>%.0f</b> now, far left %.0f before and %.0f now.' % (s['half'], s2['half'], s['left'], s2['left'])
        R = [B.row('a', 'As it came in', 'the file stops while the sound still plays' if form == 'cut' else '4.5 seconds from the middle of a longer sound', f['wet'])]
        if o['shown']: R.append(B.row('a', 'Build 0.4.4, Amount half way left', "the old model's own answer", f['o_half']))
        if n['shown']: R.append(B.row('a', 'Build 0.4.5, Amount half way left', "the new model's own answer", f['n_half']))
        if o['shown']: R.append(B.row('a', 'Build 0.4.4, Amount at far left', 'bolder', f['o_left']))
        if n['shown']: R.append(B.row('a', 'Build 0.4.5, Amount at far left', 'bolder', f['n_left']))
        R.append(B.row('ref', 'For comparison', 'the same stretch before the reverb was added', f['ref']))
        parts.append('<section class="panel"><h3>%s</h3><div class="meta">%s</div><div class="rows">%s</div></section>' % (E(title), meta, ''.join(R)))
        info.append(dict(case=case, form=form, how=how, old=dict(shown=o['shown'], p=o['p'], sure=o['sure'], first=o['first'], found=o['found'], **s), new=dict(shown=n['shown'], p=n['p'], sure=n['sure'], first=n['first'], found=n['found'], **s2), drr=drr,
                         check=dict(old=a44[case][form], new=a45[case][form])))
        print(tag, case, form, how, '| 0.4.4 shown %d p %.2f half %.1f left %.1f | 0.4.5 shown %d p %.2f half %.1f left %.1f | drr %.1f' % (o['shown'], o['p'], s['half'], s['left'], n['shown'], n['p'], s2['half'], s2['left'], drr), flush=True)
    json.dump(info, open(os.path.join(HERE, 'cut045_info.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'cut045_panels.html'), 'w').write('\n'.join(parts))
    print('panels written,', len(parts), 'examples')
