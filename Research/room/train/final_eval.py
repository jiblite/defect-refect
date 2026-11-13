# The whole reverb engine on the scoreboard, as the app will run it (engine.py): reverb out of one sound, half, doubled
# (two ways), the found room put on a second sound, and the room numbers against the truth.
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, common as cm, truth, score, engine, render, rc_look, param_ceiling as pc
from multiprocessing import Pool
corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); MODEL = os.environ.get('MODEL', 'unet_b9'); ORACLE = os.environ.get('ORACLE', '') == '1'
def one(d):
    c = truth.load(d, 'A'); direct, rev = truth.parts(c); wet = c['wet']; ft = cm.features(wet, direct); gt = np.clip(ft['LS'] - ft['LE'], -30.0, 0.0)       # gt: the right answer on the band grid
    an = engine.analyse(wet, MODEL, g=gt if ORACLE else None); sc0 = score.Scorer(direct); dw = sc0(wet); out = {}
    def sc(y): return 100.0 * (1.0 - sc0(y) / max(dw, 1e-9))
    out['reverb out'] = sc(engine.one_sound(wet, an, 0.0)); out['half way'] = sc(engine.one_sound(wet, an, 0.5)); out['bolder'] = sc(engine.fader(wet, an, 0.0))       # the app's fader: half way left, a quarter left, far left
    tgt = direct + 2.0 * rev; sc2 = score.Scorer(tgt); d2 = sc2(wet)
    out['doubled'] = 100.0 * (1.0 - sc2(engine.one_sound(wet, an, 2.0)) / max(d2, 1e-9))
    cB = truth.load(d, 'B'); sb = score.Scorer(cB['wet']); base = sb(cB['dry']); out['room onto second sound'] = 100.0 * (1.0 - sb(engine.onto(cB['dry'], an, 1.0)) / max(base, 1e-9)); bases = dict(out=float(dw), onto=float(base))
    gd = render.render(direct, an['g'], 0.0); gr = render.render(rev, an['g'], 0.0)       # the same gains on the two true parts alone: how much of each is left
    E = 10 ** (ft['LE'] / 10); live = ft['LE'] > ft['LE'].max() - 50.0; room = live & (ft['LR'] - ft['LS'] > 6.0); body = live & (ft['LS'] - ft['LR'] > 6.0); G = 10 ** (an['g'] / 10)
    cutdb = lambda sel, G=G: float(-10 * np.log10(max((E * G)[sel].sum(), 1e-30) / max(E[sel].sum(), 1e-30))) if sel.any() else float('nan')
    Gb = 10 ** (engine.deeper(an['g'], engine.DEEP) / 10)
    m = c['meta']; ed = (direct ** 2).sum(); er = (rev ** 2).sum(); mid = (cm.FC41 > 450) & (cm.FC41 < 7000); r = dict(name=c['name'], split=m['split'], kind=corp[m['A']]['kind'], drr=float(10 * np.log10(max(ed, 1e-20) / max(er, 1e-20))), found=bool(an['found']), out=out, bases=bases, rev_cut=float(10 * np.log10(max(er, 1e-20) / max((gr ** 2).sum(), 1e-20))), dir_cut=float(10 * np.log10(max(ed, 1e-20) / max((gd ** 2).sum(), 1e-20))), tail_cut=cutdb(room), body_cut=cutdb(body), tail_cut_bold=cutdb(room, Gb), body_cut_bold=cutdb(body, Gb), room_share=float(room.sum() / max(live.sum(), 1)))
    if an['found']:
        kt = 10 * np.log10((10 ** (ft['LR'] / 10)).sum(0) / np.maximum((10 ** (ft['LS'] / 10)).sum(0), 1e-30) + 1e-9); Tc = np.array([pc.t60_of(x) for x in rc_look.click_W(c)]); okb = (10 ** (ft['LS'] / 10)).max(0) > 10 ** ((ft['LE'].max() - 40.0) / 10); mm = mid & okb & np.isfinite(Tc)
        if mm.any(): r['kerr'] = float(np.median((an['kdb'] - kt)[mm])); r['terr'] = float(np.median(np.log2(an['T60'][mm] / Tc[mm])))
    return r
if __name__ == '__main__':
    t0 = time.time(); ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[::int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    with Pool(int(os.environ.get('NP', '2'))) as p: rows = p.map(one, ds)
    json.dump(rows, open('/tmp/claude-0/v2/train/final_eval_%s.json' % ('oracle' if ORACLE else MODEL), 'w')); keys = list(rows[0]['out'].keys()); drr = np.array([r['drr'] for r in rows]); sp = np.array([r['split'] for r in rows])
    print('model %s | %d cases | a room was found on %.0f%%' % ('THE RIGHT ANSWER ON THE BAND GRID' if ORACLE else MODEL, len(rows), 100 * np.mean([r['found'] for r in rows])))
    ke = np.array([r['kerr'] for r in rows if 'kerr' in r]); te = np.array([r['terr'] for r in rows if 'terr' in r])
    print('room numbers: strength bias %+.1f dB, median |error| %.1f dB, within 2 dB %.0f%% | decay time bias x%.2f, median off x%.2f, within 20%% %.0f%%' % (np.median(ke), np.median(np.abs(ke)), 100 * np.mean(np.abs(ke) < 2), 2 ** np.median(te), 2 ** np.median(np.abs(te)), 100 * np.mean(np.abs(te) < np.log2(1.2))))
    rc_ = np.array([r['rev_cut'] for r in rows]); dc_ = np.array([r['dir_cut'] for r in rows]); print('reverb part turned down by %.1f dB (median; quarter points %.1f .. %.1f), direct part loses %.2f dB (median; worst tenth %.2f)' % (np.median(rc_), np.percentile(rc_, 25), np.percentile(rc_, 75), np.median(dc_), np.percentile(dc_, 90)))
    pool = lambda key, bk: 100.0 * (1.0 - sum(r['bases'][bk] * (1.0 - r['out'][key] / 100.0) for r in rows) / sum(r['bases'][bk] for r in rows))
    print('all cases pooled (how much of the whole footprint is gone): reverb out %.1f, half way %.1f, room onto second sound %.1f' % (pool('reverb out', 'out'), pool('half way', 'out'), pool('room onto second sound', 'onto')) + ', bolder %.1f' % pool('bolder', 'out'))
    tc_ = np.array([r['tail_cut'] for r in rows]); bc_ = np.array([r['body_cut'] for r in rows]); print('where the room is on its own it comes down by %.1f dB (median; quarter points %.1f .. %.1f); where the sound itself is on top it loses %.2f dB (median; worst tenth %.2f)' % (np.nanmedian(tc_), np.nanpercentile(tc_, 25), np.nanpercentile(tc_, 75), np.nanmedian(bc_), np.nanpercentile(bc_, 90)))
    print('far left (bolder): the room on its own comes down by %.1f dB, the sound on top loses %.2f dB (medians)' % (np.nanmedian([r['tail_cut_bold'] for r in rows]), np.nanmedian([r['body_cut_bold'] for r in rows])))
    for lab, sel in [('all', np.ones(len(rows), bool)), ('reverbs also used in training', sp == 'dev'), ('reverbs and sounds never seen', sp == 'test'), ('light (DRR >= 8)', drr >= 8), ('middle', (drr < 8) & (drr >= 0)), ('wet (DRR < 0)', drr < 0)]:
        if sel.sum(): print('%-30s n=%3d | ' % (lab, sel.sum()) + ' | '.join('%s: median %5.1f, mean %5.1f, below zero %2.0f%%' % (k, np.median([r['out'][k] for r, s in zip(rows, sel) if s]), np.mean([r['out'][k] for r, s in zip(rows, sel) if s]), 100 * np.mean([r['out'][k] < 0 for r, s in zip(rows, sel) if s])) for k in keys))
    print('%.0f s' % (time.time() - t0))
