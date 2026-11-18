# The control for long_score.py: the same clip with the same reverb setting as each looped long sound (prep_long.py, loopr),
# played ONCE and ringing out (9 s) instead of six times over (39 s), through the app in the same way. Says whether a
# reverb comes out worse because the sound is long, or because of the clip and the setting.
#   python3 long_once.py path/to/drrender path/to/Resources tag part of     -> long_once_<tag>_<part>.jsonl (resumable)
#   python3 long_once.py --show tag            (pairs with long_score_<tag>_*.jsonl)
import sys, os, json, glob, re, subprocess, tempfile
sys.dont_write_bytecode = True; sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/roomsure'); sys.path.insert(0, '/home/claude/defect-refect/Research/room/bench')
import numpy as np, soundfile as sf, scipy.signal as ss
D = '/tmp/claude-0/v2/roomsure/'; SR = 48000; CLICK = 480
def show(tag):
    A = {json.loads(l)['name']: json.loads(l) for f in sorted(glob.glob(D + 'long_once_%s_*.jsonl' % tag)) for l in open(f)}
    B = {json.loads(l)['name']: json.loads(l) for f in sorted(glob.glob(D + 'long_score_%s_*.jsonl' % tag)) for l in open(f) if json.loads(l)['form'] == 'rings out'}
    K = sorted(k for k in A if k in B); print('%d pairs (the clip once, 9 s | six times over, 39 s)' % len(K))
    for lab, sel in (('all', lambda k: True), ('direct over reverb under 5 dB', lambda k: A[k]['drr'] < 5), ('direct over reverb 5 dB and more', lambda k: A[k]['drr'] >= 5)):
        g = [k for k in K if sel(k)]
        if not g: continue
        for nm, X in (('once', A), ('six times', B)):
            o = np.array([X[k]['out'] for k in g]); b = np.array([X[k]['bold'] for k in g]); sh = np.array([X[k]['shown'] for k in g], bool)
            print('%-34s %-9s n=%2d shown %2d | half way left: mean %5.1f, where shown median %5.1f, worse %2d, lowest %6.1f | far left: mean %5.1f, where shown median %5.1f, worse %2d, lowest %6.1f' % (
                lab, nm, len(g), sh.sum(), o.mean(), np.median(o[sh]) if sh.any() else 0, (o < 0).sum(), o.min(), b.mean(), np.median(b[sh]) if sh.any() else 0, (b < 0).sum(), b.min()))
    both = [k for k in K if A[k]['shown'] and B[k]['shown']]; d = np.array([B[k]['out'] - A[k]['out'] for k in both]); e = np.array([B[k]['bold'] - A[k]['bold'] for k in both])
    print('shown both ways: %d; six times minus once, half way left: median %+.1f (quartiles %+.1f, %+.1f); far left: median %+.1f (quartiles %+.1f, %+.1f)' % (len(both), np.median(d), np.percentile(d, 25), np.percentile(d, 75), np.median(e), np.percentile(e, 25), np.percentile(e, 75)))
    if '-v' in sys.argv:
        for k in sorted(K, key=lambda k: B[k]['out']): print('  %-34s drr %5.1f | once: shown %d half %6.1f far %7.1f | six times: shown %d half %6.1f far %7.1f' % (k, A[k]['drr'], A[k]['shown'], A[k]['out'], A[k]['bold'], B[k]['shown'], B[k]['out'], B[k]['bold']))
if sys.argv[1] == '--show': show(sys.argv[2]); sys.exit(0)
import zoo as R, prep_long as PL, score
EXE, RES, TAG = sys.argv[1:4]; part, of = int(sys.argv[4]), int(sys.argv[5])
OUT = D + 'long_once_%s_%d.jsonl' % (TAG, part); done = {json.loads(l)['name'] for l in open(OUT)} if os.path.exists(OUT) else set(); tmp = tempfile.mkdtemp(); W = D + 'wav/long_once/'
def sc(direct, wet, y): s = score.Scorer(direct); return 100.0 * (1.0 - s(y) / max(s(wet), 1e-9))
T = [r for r in json.load(open(D + 'long_truth.json')) if r['kind'] == 'loopr']
for r in T[part::of]:
    if r['name'] in done: continue
    name = r['tool']; i = int(r['name'][-2:]); rng = PL.rng_of('loopr', name, i); a = str(rng.choice(PL.NAMES[r['split']])); x = PL.dry(a); p = R.ZOO[name][2](rng)
    w = R.apply(name, x, p, 3.0); pk = np.abs(w).max(); g = 0.98 / pk if pk >= 0.98 else 1.0; n = len(w)
    cl = R.impulse(SR * 6, CLICK); cl[:, 1] = 0; hl = R.apply(name, cl, p)[CLICK:]; cr = R.impulse(SR * 6, CLICK); cr[:, 0] = 0; hr = R.apply(name, cr, p)[CLICK:]
    e = (hl ** 2).sum(1) + (hr ** 2).sum(1); k = int(np.argmax(e)); a0, b0 = max(0, k - 48), k + 120
    direct = np.stack([ss.fftconvolve(x[:, 0], hl[a0:b0, o])[:n] + ss.fftconvolve(x[:, 1], hr[a0:b0, o])[:n] for o in range(2)], 1)
    if a0 > 0: direct = np.vstack([np.zeros((a0, 2)), direct])[:n]
    if len(direct) < n: direct = np.vstack([direct, np.zeros((n - len(direct), 2))])
    rev = w - direct; drr = 10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20))
    pi = W + r['name'] + '.wav'; sf.write(pi, w * g, SR, subtype='PCM_16'); wet = sf.read(pi, dtype='float64', always_2d=True)[0]; dd = direct * g
    p1, p2 = (os.path.join(tmp, v) for v in ('half.wav', 'left.wav'))
    q = subprocess.run([EXE, RES, pi, '-', p1, 'reverb.amount=0.5', '--', '-', p2, 'reverb.amount=0'], capture_output=True, text=True)
    if q.returncode: raise RuntimeError(q.stdout[-300:] + q.stderr[-300:])
    gg = dict((u, float(v)) for u, v in re.findall(r'^([a-z0-9]+) p=([0-9.eE+-]+)', q.stdout, re.M)); rm = re.search(r'^room: found=(\d) .*? sureness=([0-9.eE+-]+)', q.stdout, re.M)
    shown = 'reverb' in gg and '(room engine)' in q.stdout; rd = lambda f: sf.read(f, dtype='float64', always_2d=True)[0][:len(wet)]
    o = dict(name=r['name'], tool=name, split=r['split'], drr=float(drr), drr_long=r['drr'], sec=n / SR, found=bool(int(rm.group(1))) if rm else False, sure=float(rm.group(2)) if rm else None, shown=bool(shown),
             others=sorted(u for u in gg if u != 'reverb'), out=sc(dd, wet, rd(p1)) if shown else 0.0, bold=sc(dd, wet, rd(p2)) if shown else 0.0)
    open(OUT, 'a').write(json.dumps(o) + '\n'); print('%-36s drr %5.1f found %d sure %s shown %d | half %6.1f far left %6.1f | others %s' % (r['name'], drr, o['found'], '%.2f' % o['sure'] if o['sure'] is not None else '-', o['shown'], o['out'], o['bold'], ','.join(o['others'])), flush=True)
