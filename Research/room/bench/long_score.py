# How well the app takes the reverb out of a LONG sound (the sounds of prep_long.py with reverb; truths by
# prep_long_truth.py), as train/cut_app.py asks it of sounds of 6 and 9 s: the app itself (drrender, default Sureness),
# REVERB Amount half way left and at far left, 100 = the true direct part, 0 = like doing nothing; a sound on which no
# REVERB group is shown stays as it is and counts 0.
#   python3 long_score.py path/to/drrender path/to/Resources tag part of [cut]     (cut: also the sound cut off, see long_rows.json)  -> long_score_<tag>_<part>.jsonl (resumable)
#   python3 long_score.py --show tag [tag2 ...]
import sys, os, json, glob, re, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/home/claude/defect-refect/Research/room/bench')
import numpy as np, soundfile as sf
D = '/tmp/claude-0/v2/roomsure/'; SR = 48000
def show(tags):
    for t in tags:
        R = [json.loads(l) for f in sorted(glob.glob(D + 'long_score_%s_*.jsonl' % t)) for l in open(f)]
        for form in sorted({r['form'] for r in R}):
            for lab, sel in (('all', lambda r: True), ('looped clip', lambda r: r['kind'] == 'loopr'), ('clips in a row', lambda r: r['kind'] == 'catr'), ('tools training used', lambda r: r['split'] == 'dev'), ('tools never seen', lambda r: r['split'] == 'test')):
                g = [r for r in R if r['form'] == form and sel(r)]
                if not g: continue
                k2 = 'half' if 'half' in g[0] else 'bold'; o = np.array([r['out'] for r in g]); b = np.array([r[k2] for r in g]); sh = np.array([r['shown'] for r in g], bool)
                print(('[%s] %-9s %-20s n=%2d shown %2d (%3.0f%%) | ' + ('far left = the answer' if k2 == 'half' else 'half way left') + ': mean %5.1f, where shown median %5.1f, worse %2d | ' + ('half way left' if k2 == 'half' else 'far left') + ': mean %5.1f, where shown median %5.1f, worse %2d') % (
                    t, form, lab, len(g), sh.sum(), 100 * sh.mean(), o.mean(), np.median(o[sh]) if sh.any() else 0, (o < 0).sum(), b.mean(), np.median(b[sh]) if sh.any() else 0, (b < 0).sum()))
if sys.argv[1] == '--show': show(sys.argv[2:]); sys.exit(0)
import score
EXE, RES, TAG = sys.argv[1:4]; part, of = int(sys.argv[4]), int(sys.argv[5]); withcut = 'cut' in sys.argv[6:]
OUT = D + 'long_score_%s_%d.jsonl' % (TAG, part); done = {(json.loads(l)['name'], json.loads(l)['form']) for l in open(OUT)} if os.path.exists(OUT) else set(); tmp = tempfile.mkdtemp()
T = json.load(open(D + 'long_truth.json'))
NEW = os.environ.get('FADER') == '046'; A1, A2, K2 = ('0', '0.5', 'half') if NEW else ('0.5', '0', 'bold')      # FADER=046: a tool of build 0.4.6 or later, where far left is the engine's own answer ('out') and half way left half of it ('half')
def sc(direct, wet, y): s = score.Scorer(direct); return 100.0 * (1.0 - s(y) / max(s(wet), 1e-9))
ONLY = [x for x in os.environ.get('ONLY', '').split(',') if x]          # ONLY=name,name in the environment: just these sounds
for r in T[part::of]:
    if ONLY and r['name'] not in ONLY: continue
    wet = sf.read(r['path'], dtype='float64', always_2d=True)[0]; direct = sf.read(r['truth'], dtype='float64', always_2d=True)[0]
    forms = [('rings out', len(wet))] + ([('cut off', int((r['dry_sec'] - 1) * SR))] if withcut else [])
    for form, n in forms:
        if (r['name'], form) in done: continue
        w, dd = wet[:n], direct[:n]; pi, p1, p2 = (os.path.join(tmp, x) for x in ('in.wav', 'half.wav', 'left.wav')); sf.write(pi, w, SR, subtype='FLOAT')
        q = subprocess.run([EXE, RES, pi, '-', p1, 'reverb.amount=' + A1, '--', '-', p2, 'reverb.amount=' + A2], capture_output=True, text=True)
        if q.returncode: raise RuntimeError(q.stdout[-300:] + q.stderr[-300:])
        g = dict((a, float(b)) for a, b in re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', q.stdout, re.M)); rm = re.search(r'^room: found=(\d) .*? sureness=([0-9.eE+-]+)', q.stdout, re.M)
        shown = 'reverb' in g and '(room engine)' in q.stdout; rd = lambda p: sf.read(p, dtype='float64', always_2d=True)[0][:len(w)]
        o = dict(name=r['name'], form=form, kind=r['kind'], tool=r['tool'], split=r['split'], drr=r['drr'], sec=n / SR, found=bool(int(rm.group(1))) if rm else False, sure=float(rm.group(2)) if rm else None, shown=bool(shown),
                 others=sorted(k for k in g if k != 'reverb'), out=sc(dd, w, rd(p1)) if shown else 0.0, base=float(score.Scorer(dd)(w)), **{K2: sc(dd, w, rd(p2)) if shown else 0.0})
        open(OUT, 'a').write(json.dumps(o) + '\n'); print('%-36s %-9s found %d sure %s shown %d | out %6.1f %s %6.1f | others %s' % (r['name'], form, o['found'], '%.2f' % o['sure'] if o['sure'] is not None else '-', o['shown'], o['out'], K2, o[K2], ','.join(o['others'])), flush=True)
