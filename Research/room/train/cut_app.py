# The app itself (Tests/render.cpp built as drrender, at the default Sureness) on the scoreboard's reverb cases in three
# forms: the sound as it rings out (9 s), CUT OFF at 6 s, and as an EXCERPT (1.0 to 5.5 s, cut at both ends). cut_eval.py
# asks the same of the engine alone; here the app decides for itself whether it shows a REVERB group at all, and a case
# where it does not is left alone (score 0), as a user with the faders untouched would get it.
#   python3 cut_app.py path/to/drrender path/to/Resources tag [step] [part of]     -> cut_app_<tag>_<part>.jsonl (resumable)
#   python3 cut_app.py --show tag [tag2]                                           (two tags: side by side)
# Per case and form: the room engine's verdict, whether REVERB is shown, and the score of the REVERB Amount fader half
# way left (the engine's own answer) and at far left (100 = the true direct part, 0 = like doing nothing).
# FADER=046 in the environment: the tool is of build 0.4.6 or later, where FAR LEFT is the engine's own answer in full
# ('out') and half way left is half of it in dB ('half'; the bolder end of the earlier builds, 'bold', is gone).
import sys, os, json, glob, re, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, soundfile as sf
HERE = '/tmp/claude-0/v2/train/'; SR = 48000; CUT = 6 * SR; E0, E1 = SR, int(5.5 * SR)
FORMS = (('whole', 'rings out, whole file'), ('cut', 'cut off at 6 s'), ('excerpt', 'excerpt 1.0 to 5.5 s'))
def show(tags):
    data = {t: [json.loads(l) for f in sorted(glob.glob(HERE + 'cut_app_%s_*.jsonl' % t)) for l in open(f)] for t in tags}
    for lab, sel in (('all', lambda r: True), ('reverbs also used in training', lambda r: r['split'] == 'dev'), ('reverbs and sounds never seen', lambda r: r['split'] == 'test')):
        print('%s (n=%d)' % (lab, len([r for r in data[tags[0]] if sel(r)])))
        for tag, what in FORMS:
            for t in tags:
                rs = [r for r in data[t] if sel(r)]
                if not rs: continue
                k2 = 'half' if 'half' in rs[0][tag] else 'bold'; o = np.array([r[tag]['out'] for r in rs]); b = np.array([r[tag][k2] for r in rs]); sh = np.array([r[tag]['shown'] for r in rs], bool)
                print(('  %-24s [%-5s] room found %3.0f%% shown %3.0f%% | ' + ('far left = the answer' if k2 == 'half' else 'reverb out') + ': median %5.1f mean %6.1f worse %2.0f%% | ' + ('half way left' if k2 == 'half' else 'far left') + ': median %5.1f mean %6.1f worse %2.0f%% | where shown (n=%2d): median %5.1f / %5.1f') % (
                    what, t, 100 * np.mean([r[tag]['found'] for r in rs]), 100 * sh.mean(), np.median(o), o.mean(), 100 * np.mean(o < 0), np.median(b), b.mean(), 100 * np.mean(b < 0), sh.sum(),
                    np.median(o[sh]) if sh.any() else 0, np.median(b[sh]) if sh.any() else 0))
                if 'base' in rs[0][tag]:
                    B = np.array([r[tag]['base'] for r in rs]); print('  %-24s [%-5s] in dB per cell: untouched %.2f from the dry sound (median) | first position: closer by 0.5 dB+ on %d, further by 0.25 dB+ on %d, by 0.5 dB+ on %d | second: closer %d, further %d, %d' % ('', t, np.median(B), (-B * o / 100 <= -0.5).sum(), (-B * o / 100 >= 0.25).sum(), (-B * o / 100 >= 0.5).sum(), (-B * b / 100 <= -0.5).sum(), (-B * b / 100 >= 0.25).sum(), (-B * b / 100 >= 0.5).sum()))
if sys.argv[1] == '--show': show(sys.argv[2:]); sys.exit(0)
import truth, score
EXE, RES, TAG = sys.argv[1:4]; step = int(sys.argv[4]) if len(sys.argv) > 4 else 3; part, of = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
NEW = os.environ.get('FADER') == '046'; A1, A2, K2 = ('0', '0.5', 'half') if NEW else ('0.5', '0', 'bold')
OUT = HERE + 'cut_app_%s_%d.jsonl' % (TAG, part); done = {json.loads(l)['name'] for l in open(OUT)} if os.path.exists(OUT) else set(); tmp = tempfile.mkdtemp()
def sc(direct, wet, y): s = score.Scorer(direct); return 100.0 * (1.0 - s(y) / max(s(wet), 1e-9))
def dist(direct, wet): return float(score.Scorer(direct)(wet))        # how far the untouched sound is from its true direct part, dB per cell of the score's pictures
for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::step][part::of]:
    c = truth.load(d, 'A')
    if c['name'] in done: continue
    direct, rev = truth.parts(c); wet = c['wet']; m = c['meta']; r = dict(name=c['name'], split=m['split'], drr=float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20))))
    for tag, lo, hi in (('whole', 0, len(wet)), ('cut', 0, CUT), ('excerpt', E0, E1)):
        w, dd = wet[lo:hi], direct[lo:hi]; pi, p1, p2 = (os.path.join(tmp, n) for n in ('in.wav', 'half.wav', 'left.wav')); sf.write(pi, w, SR, subtype='FLOAT')
        q = subprocess.run([EXE, RES, pi, '-', p1, 'reverb.amount=' + A1, '--', '-', p2, 'reverb.amount=' + A2], capture_output=True, text=True)
        if q.returncode: raise RuntimeError(q.stdout[-300:] + q.stderr[-300:])
        g = dict((a, float(b)) for a, b in re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', q.stdout, re.M)); rm = re.search(r'^room: found=(\d) .*? sureness=([0-9.eE+-]+)', q.stdout, re.M)
        shown = 'reverb' in g and '(room engine)' in q.stdout; rd = lambda p: sf.read(p, dtype='float64', always_2d=True)[0][:len(w)]
        r[tag] = dict(found=bool(int(rm.group(1))) if rm else False, sure=float(rm.group(2)) if rm else None, p=g.get('reverb'), shown=bool(shown), others=sorted(k for k in g if k != 'reverb'),
                      out=sc(dd, w, rd(p1)) if shown else 0.0, base=dist(dd, w), **{K2: sc(dd, w, rd(p2)) if shown else 0.0})
    open(OUT, 'a').write(json.dumps(r) + '\n'); print(r['name'], ' | '.join('%s: found %d shown %d p %s out %6.1f %s %6.1f' % (t, r[t]['found'], r[t]['shown'], '%.2f' % r[t]['p'] if r[t]['p'] is not None else '-', r[t]['out'], K2, r[t][K2]) for t, _ in FORMS), flush=True)
