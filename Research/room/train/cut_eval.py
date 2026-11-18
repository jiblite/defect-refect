# The whole reverb engine (engine.py, as the app runs it) on the scoreboard's reverb cases in three forms: the sound as it
# rings out (9 s), CUT OFF at 6 s, and as an EXCERPT (1.0 to 5.5 s, cut at both ends). Scores how well the reverb comes
# out (100 = the true direct part, 0 = like doing nothing), half way and far left, and whether a room is found at all.
# The ringing form is scored twice: over the whole file and over its first 6 s only, so that the cut-off form can be
# compared over the same stretch.
#   MODEL=unet_w+unet_wp2 NP=1 python3 cut_eval.py [step]      -> cut_eval_<MODEL>.json
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, truth, score, engine
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_w+unet_wp2'); SR = 48000; CUT = 6 * SR; E0, E1 = SR, int(5.5 * SR)
def sc(direct, wet, y): s = score.Scorer(direct); return 100.0 * (1.0 - s(y) / max(s(wet), 1e-9))
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; m = c['meta']; r = dict(name=c['name'], split=m['split'], drr=float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20))))
    for tag, lo, hi in (('whole', 0, len(wet)), ('cut', 0, CUT), ('excerpt', E0, E1)):
        w, dd = wet[lo:hi], direct[lo:hi]; an = engine.analyse(w, MODEL); y = engine.one_sound(w, an, 0.0); yb = engine.fader_bold(w, an, 0.0)
        r[tag] = dict(found=bool(an['found']), out=sc(dd, w, y), bold=sc(dd, w, yb))
        if tag == 'whole': r['whole6'] = dict(found=bool(an['found']), out=sc(dd[:CUT], w[:CUT], y[:CUT]), bold=sc(dd[:CUT], w[:CUT], yb[:CUT])); r['wholeE'] = dict(found=bool(an['found']), out=sc(dd[E0:E1], w[E0:E1], y[E0:E1]), bold=sc(dd[E0:E1], w[E0:E1], yb[E0:E1]))
    return r
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '1'))) as p: rows = p.map(one, ds)
    json.dump(rows, open('/tmp/claude-0/v2/train/cut_eval_%s.json' % MODEL, 'w')); print('model %s | %d cases' % (MODEL, len(rows)))
    for lab, sel in (('all', lambda r: True), ('reverbs also used in training', lambda r: r['split'] == 'dev'), ('reverbs and sounds never seen', lambda r: r['split'] == 'test')):
        rs = [r for r in rows if sel(r)]
        if not rs: continue
        print('%s (n=%d)' % (lab, len(rs)))
        for tag, what in (('whole', 'rings out, whole file'), ('whole6', 'rings out, its first 6 s'), ('cut', 'cut off at 6 s'), ('wholeE', 'rings out, 1.0 to 5.5 s of it'), ('excerpt', 'excerpt 1.0 to 5.5 s')):
            o = np.array([r[tag]['out'] for r in rs]); b = np.array([r[tag]['bold'] for r in rs])
            print('  %-30s room found %3.0f%% | reverb out: median %5.1f mean %6.1f below zero %2.0f%% | far left: median %5.1f mean %6.1f below zero %2.0f%%' % (what, 100 * np.mean([r[tag]['found'] for r in rs]), np.median(o), np.mean(o), 100 * np.mean(o < 0), np.median(b), np.mean(b), 100 * np.mean(b < 0)))
    print('%.0f s' % (time.time() - t0))
