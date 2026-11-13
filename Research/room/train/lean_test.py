# How loud should the found room go on the second sound? Scores of "room onto second sound" with the found room turned
# up or down by a few dB, per case (to see whether leaning dry pays, and for which rooms).
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, score, engine, roomcap as rc
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_b9'); ORACLE = os.environ.get('ORACLE', '') == '1'; DS = (-9.0, -6.0, -4.0, -2.0, 0.0, 2.0)
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; ft = cm.features(wet, direct); gt = np.clip(ft['LS'] - ft['LE'], -30.0, 0.0); an = engine.analyse(wet, MODEL, g=gt if ORACLE else None)
    r = dict(name=c['name'], found=bool(an['found']), drr=float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20))))
    if not an['found']: return r
    cB = truth.load(d, 'B'); sb = score.Scorer(cB['wet']); base = sb(cB['dry']); h = engine.room_ir(an); tail = rc.apply_room(cB['dry'], h, 1.0) - cB['dry']; mid = (cm.FC41 > 450) & (cm.FC41 < 7000)
    r.update(t60=float(np.median(an['T60'][mid])), kdb=float(np.median(an['kdb'][mid])), sc={str(x): float(100.0 * (1.0 - sb(cB['dry'] + 10 ** (x / 20.0) * tail) / max(base, 1e-9))) for x in DS}); return r
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '2'))) as p: rows = p.map(one, ds)
    tag = 'oracle' if ORACLE else MODEL; json.dump(rows, open('/tmp/claude-0/v2/train/lean_test_%s.json' % tag, 'w')); ok = [r for r in rows if r['found']]
    print('mask: %s | %d cases with a room' % (tag, len(ok)))
    for lab, sel in [('all', lambda r: True), ('decay under 0.6 s', lambda r: r['t60'] < 0.6), ('decay 0.6 to 1.5 s', lambda r: 0.6 <= r['t60'] < 1.5), ('decay over 1.5 s', lambda r: r['t60'] >= 1.5), ('light (DRR >= 8)', lambda r: r['drr'] >= 8), ('middle', lambda r: 0 <= r['drr'] < 8), ('wet (DRR < 0)', lambda r: r['drr'] < 0)]:
        s = [r for r in ok if sel(r)]
        if s: print('%-20s n=%2d | ' % (lab, len(s)) + ' | '.join('%+.0f dB: med %5.1f mean %5.1f' % (x, np.median([r['sc'][str(x)] for r in s]), np.mean([r['sc'][str(x)] for r in s])) for x in DS))
    print('%.0f s' % (time.time() - t0))
