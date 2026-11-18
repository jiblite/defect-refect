# The page's section for build 0.4.6 (the reverb Amount fader: far left is now the engine's own answer in full, the bolder
# end of the earlier builds is gone; the room model trained again with a loss that is careful under the sound; a guard on
# its second look): four test reverbs, rendered by the app's own processing (the command-line tool drrender) as build
# 0.4.6 has it (Amount at far left and half way left) and as build 0.4.5 had it (its own answer at half way left, its
# bolder far left), as MP3s in audio/fd*_*.mp3, and the examples' HTML in fader046_panels.html (build.py puts
# fader046_head.html and these last).
#   python3 fader046_section.py
# The examples are picked by fixed rules (see pick). Scores: 100 = the sound before the reverb, 0 = no better than
# untouched, over the whole file and over the sound itself (the first 6 s, without the ring-out; train/depth2_test.py).
import os, sys, re, json, html, glob, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/train'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.environ.setdefault('OMP_NUM_THREADS', '2')
import truth, build as B, depth2_test as D
SR = 48000; TMP = '/tmp/claude-0/v2/fader046/'; os.makedirs(TMP, exist_ok=True); SETS = '/tmp/claude-0/v2/bench/sets/reverb/'; E = html.escape; FIRST = int(os.environ.get('FIRST', '29'))
OLD = (os.environ.get('OLD', '/tmp/claude-0/v2/bench/drrender_head'), os.environ.get('OLD_RES', '/tmp/claude-0/v2/res_045'))                     # build 0.4.5: its code AND its weights
NEW = (os.environ.get('NEW', '/tmp/claude-0/v2/bench/drrender_046b'), os.environ.get('NEW_RES', '/home/claude/defect-refect/Resources'))
def rows_of(tag): return {r['name']: r for f in sorted(glob.glob('/tmp/claude-0/v2/train/cut_app_%s_*.jsonl' % tag)) for r in map(json.loads, open(f))}
def app(exe_res, path, n, amounts):
    """The app on one file with the REVERB Amount fader at each of the given positions: (shown, sureness, [sound per position])."""
    ps = [TMP + 'o%d.wav' % i for i in range(len(amounts))]; args = [exe_res[0], exe_res[1], path]
    for i, a in enumerate(amounts): args += (['--'] if i else []) + ['-', ps[i], 'reverb.amount=%g' % a]
    q = subprocess.run(args, capture_output=True, text=True)
    if q.returncode: raise RuntimeError(q.stdout[-300:] + q.stderr[-300:])
    shown = re.search(r'^reverb p=([0-9.eE+-]+) \(room engine\)', q.stdout, re.M); return bool(shown), float(shown.group(1)) if shown else 0.0, [sf.read(p, dtype='float64', always_2d=True)[0][:n] for p in ps]
def pick():
    """Fixed rules. Three from the 40 test reverbs of cut_app.py as build 0.4.5 scored them (the file that rings out): the
    cases on which REVERB is shown, ranked by how much the bolder far left of 0.4.5 scored against its plain answer; the
    first (the bolder end did most harm), the middle one, and the last (the bolder end was best). The fourth from all 120
    test reverbs (train/depth2_test.py, engine alone): the one whose sound itself the plain answer of build 0.4.5 moved
    furthest from the sound before the reverb."""
    a = rows_of('045b'); S = sorted((n for n in a if a[n]['whole']['shown']), key=lambda n: a[n]['whole']['bold'] - a[n]['whole']['out'])
    P = [(S[0], 'where the bolder end did most harm (place 1 of %d)' % len(S)), (S[len(S) // 2 - 1], 'a typical case (place %d of %d)' % (len(S) // 2, len(S))), (S[-1], 'where the bolder end did best (place %d of %d)' % (len(S), len(S)))]
    rows = json.load(open('/tmp/claude-0/v2/train/depth2_test_unet_x+unet_xp2_e5.json')); w = max(rows, key=lambda r: -r['base'][1] * r['sc']['answer'][1])
    return P + [(w['name'], 'the one of all 120 test reverbs on which the answer of build 0.4.5 did most harm to the sound itself')]
if __name__ == '__main__':
    parts = []; info = []
    for i, (case, how) in enumerate(pick()):
        c = truth.load(SETS + case + '/', 'A'); direct, rev = truth.parts(c); m = c['meta']; w = c['wet']; n = len(w); L = 6 * SR; tag = 'fd%d' % (i + 1); p = TMP + tag + '.wav'; sf.write(p, w, SR, subtype='FLOAT')
        so, po, (bold, oldans) = app(OLD, p, n, [0.0, 0.5]); sn, pn, (full, half) = app(NEW, p, n, [0.0, 0.5])
        Cw = D.cells(direct, w); base = [D.seg(Cw, 0, n), D.seg(Cw, 0, L)]
        def sc(y): C = D.cells(direct, y); return [100.0 * (1.0 - D.seg(C, 0, n) / max(base[0], 1e-9)), 100.0 * (1.0 - D.seg(C, 0, L) / max(base[1], 1e-9))]         # whole file, the sound itself
        s = dict(full=sc(full), half=sc(half), oldans=sc(oldans), bold=sc(bold)); a0 = max(0, int((m.get('pre', 0.0) - 0.15) * SR))        # the test files start with silence: not needed for listening
        x = dict(wet=w, full=full, half=half, oldans=oldans, bold=bold, ref=direct); g = 0.89 / max(np.abs(v).max() for v in x.values()); f = {k: B.mp3('%s_%s' % (tag, k), v[a0:], g) for k, v in x.items()}
        drr = float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max(((w - direct) ** 2).sum(), 1e-20)))
        title = '%d. %s through %s' % (FIRST + i, B.cap(B.sound_name(m['A'])), B.EFFECT[m['effect']])
        new = ('REVERB shown at %.0f %% sure' % (100 * pn)) if sn else 'Build 0.4.6 does not show REVERB on this file at the default Sureness (%.0f %% sure), so it would stay as it is; the two rows of build 0.4.6 are what it gives with Sureness turned down' % (100 * pn)
        meta = ('%s; %s. %s (build 0.4.5: %.0f %%). Scores (100 = the sound before the reverb, 0 = no better than untouched), whole file and, in brackets, the sound itself without its ring-out: build 0.4.6 at far left <b>%.0f</b> (%.0f), half way left %.0f (%.0f); '
                'build 0.4.5 at half way left %.0f (%.0f), at far left %.0f (%.0f).') % (E(B.cap(B.wetness(drr))), E(how), new, 100 * po, s['full'][0], s['full'][1], s['half'][0], s['half'][1], s['oldans'][0], s['oldans'][1], s['bold'][0], s['bold'][1])
        R = [B.row('a', 'As it came in', 'with the reverb on it', f['wet']), B.row('a', 'Build 0.4.6, far left', "the app's own answer in full", f['full']), B.row('a', 'Build 0.4.6, half way left', 'half of it', f['half']),
             B.row('a', 'Build 0.4.5, half way left', 'the answer of the model before this one', f['oldans']), B.row('a', 'Build 0.4.5, far left', 'the bolder end that is gone', f['bold']),
             B.row('ref', 'For comparison', 'the sound before the reverb was added', f['ref'])]
        parts.append('<section class="panel"><h3>%s</h3><div class="meta">%s</div><div class="rows">%s</div></section>' % (E(title), meta, ''.join(R)))
        info.append(dict(case=case, how=how, drr=drr, shown_old=so, shown_new=sn, p_old=po, p_new=pn, base=base, **s))
        print(tag, case, how, '| shown %d %d p %.2f %.2f | 0.4.6 far left %.1f (%.1f) half %.1f (%.1f) | 0.4.5 half way %.1f (%.1f) far left %.1f (%.1f) | sound itself from dry %.2f dB | drr %.1f'
              % (so, sn, po, pn, s['full'][0], s['full'][1], s['half'][0], s['half'][1], s['oldans'][0], s['oldans'][1], s['bold'][0], s['bold'][1], base[1], drr), flush=True)
    json.dump(info, open(os.path.join(HERE, 'fader046_info.json'), 'w'), indent=1); open(os.path.join(HERE, 'fader046_panels.html'), 'w').write('\n'.join(parts)); print('panels written,', len(parts), 'examples')
