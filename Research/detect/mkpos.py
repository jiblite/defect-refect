# Builds the detection scoreboard: for every effect family the app names, cases made with OUTSIDE effects (zoo_fx.py) on dry
# corpus clips, stored like the reverb scoreboard's cases: <family>/<effect>_<nn>/ with meta.json, A_wet.flac, B_wet.flac (two sounds
# through the same settings; 24-bit FLAC, gain in meta), ir_L.flac, ir_R.flac (the effect's answer to a click on one side).
# The dry sounds are the corpus clips named in meta (A, B). Half of each family's cases use effects and clips of the
# 'dev' split, half of the 'test' split.     python3 mkpos.py [family ...]
import sys, os, json, hashlib
import paths
import numpy as np, soundfile as sf, scipy.signal as ss
import zoo_fx as Z
from cpuslot import slot
SR = 48000; CORPUS = paths.CORPUS; OUT = paths.SETS; CLICK = 480
FAMILIES = {
 'flanger': [('flg_calf', 2), ('flg_sox', 2), ('flg_swh', 2), ('flg_mda_thruzero', 2)],
 'chorus': [('cho_calf', 2), ('cho_sox', 2), ('cho_pedalboard', 2), ('cho_tap', 2)],
 'phaser': [('pha_calf', 2), ('pha_swh', 2), ('pha_pedalboard', 2), ('pha_caps', 2)],
 'haas': [('haas_sox', 2), ('haas_calf_stereotools', 2), ('haas_ffmpeg', 2), ('haas_mda', 2)],
 'width': [('wid_caps', 2), ('wid_calf_haasenh', 2), ('wid_mda_comb', 2), ('wid_calf_base', 2)],
 'delay': [('dly_calf', 2), ('dly_sox', 2), ('dly_pedalboard', 2), ('dly_tap', 2)],
 'distortion': [('dist_pedalboard', 2), ('dist_sox', 2), ('dist_caps', 2), ('dist_mda', 2)],
 'bitcrush': [('bit_calf', 2), ('bit_pedalboard', 2), ('bit_swh', 2), ('bit_mda', 2)],
 'filtersweep': [('swp_pb_ladder', 2), ('swp_pb_bandpass', 2), ('swp_mda', 2), ('swp_caps', 2)],
 'tremolo': [('trm_sox', 2), ('trm_calf', 2), ('trm_tap', 2), ('trm_gx', 2)],
 'autopan': [('pan_tap', 2), ('pan_calf', 2), ('pan_sox', 2), ('pan_mda', 2)],
 'saturation': [('sat_calf', 2), ('sat_tap_tube', 2), ('sat_swh_valve', 2), ('sat_calf_tape', 2)],
 'exciter': [('exc_calf', 2), ('exc_sox', 2), ('exc_caps_spice', 4)],
 'ringmod': [('rng_calf', 2), ('rng_sox', 2), ('rng_swh', 2), ('rng_mda', 2)],
 'filter': [('fil_calf', 2), ('fil_sox', 2), ('fil_pb_ladder', 2), ('fil_swh_ls', 2)],
 'stutter': [('stu_sox', 4), ('stu_ffmpeg', 4)],
 'gatedrhythm': [('gat_sox', 2), ('gat_calf', 2), ('gat_ffmpeg_keyed', 4)],
}
meta = json.load(open(CORPUS + 'meta.json'))
def dry(name): return sf.read(CORPUS + 'dry/' + name + '.wav', always_2d=True)[0].astype(np.float64)
def wflac(path, x, g=None):
    pk = np.abs(x).max()
    if g is None: g = 1.0 if pk < 0.98 else 0.98 / pk
    sf.write(path, x * g, SR, format='FLAC', subtype='PCM_24'); return g
def db(a, b): return float(10 * np.log10((a ** 2).sum() / max((b ** 2).sum(), 1e-30) + 1e-30))
def change_db(x, w):
    """How far the wet sound is from the dry one after the best level match (dB against the dry sound): very low = the effect did nothing audible here."""
    n = len(x); g = (w[:n] * x).sum() / ((x ** 2).sum() + 1e-30); return db(w[:n] - g * x, x * (g if abs(g) > 1e-3 else 1.0)) if abs(g) > 1e-3 else 99.0
def case(name, i, family, tail, seed=7):
    fam, split, sampler, fn, clipf = Z.ZOO[name]; assert fam == family
    d = '%s%s/%s_%02d/' % (OUT, family, name, i)
    if os.path.exists(d + 'meta.json'): return 'kept'
    rng = np.random.default_rng(int(hashlib.md5((name + str(seed) + '/%d' % i).encode()).hexdigest(), 16) % 2**32); names = sorted(n for n, m in meta.items() if m['split'] == split)
    if clipf is not None: names = [n for n in names if clipf(dry(n))]            # e.g. raising the side needs a sound that has one
    for tries in range(1, 9):
        p = sampler(rng); a, b = (str(v) for v in rng.choice(names, 2, replace=False)); xa, xb = dry(a), dry(b)
        if clipf is not None and not (clipf(xa) and clipf(xb)): continue
        wa = Z.apply(name, xa, p, tail); wb = Z.apply(name, xb, p, tail)
        if not (np.isfinite(wa).all() and np.isfinite(wb).all()): continue
        ca, cb = change_db(xa, wa), change_db(xb, wb)
        if min(ca, cb) > -30 and min(np.abs(wa).max(), np.abs(wb).max()) > 1e-3: break           # the effect is there to be heard on both sounds
    else: return 'FAILED: no audible case in 8 draws'
    os.makedirs(d, exist_ok=True); ir = True
    try:
        cl = impulse = np.zeros((SR * 6, 2)); cl[CLICK, 0] = 1.0; hl = Z.apply(name, cl, p); cr = np.zeros((SR * 6, 2)); cr[CLICK, 1] = 1.0; hr = Z.apply(name, cr, p)
        if not (np.isfinite(hl).all() and np.isfinite(hr).all()): raise ValueError('nan')
    except Exception: hl = hr = np.zeros((SR * 6, 2)); ir = False
    N = len(wa); pred = np.stack([ss.fftconvolve(xa[:, 0], hl[CLICK:, c])[:N] + ss.fftconvolve(xa[:, 1], hr[CLICK:, c])[:N] for c in range(2)], 1)
    ga = wflac(d + 'A_wet.flac', wa); gb = wflac(d + 'B_wet.flac', wb); pk = max(np.abs(hl).max(), np.abs(hr).max(), 1e-9); gi = 1.0 if pk < 0.98 else 0.98 / pk
    wflac(d + 'ir_L.flac', hl, gi); wflac(d + 'ir_R.flac', hr, gi)
    json.dump(dict(effect=name, family=family, split=split, params=p, A=a, B=b, tail=tail, pre=Z.PRE, click=CLICK, gainA=ga, gainB=gb, gainIR=gi, fixed_db=db(wa - pred, wa),
                   change_db=[ca, cb], draws=tries, ir_valid=ir,
                   note='fixed_db: how well the click answers describe what happened to sound A (only a fixed linear effect is described by them). change_db: wet against dry after level match.'),
              open(d + 'meta.json', 'w'), indent=1)
    return 'built (%d draws, change %.0f / %.0f dB)' % (tries, ca, cb)
if __name__ == '__main__':
    for fam in (sys.argv[1:] or list(FAMILIES)):
        for name, n in FAMILIES[fam]:
            for i in range(n):
                with slot():
                    try: r = case(name, i, fam, Z.TAIL.get(fam, 0.0))
                    except Exception as e: r = 'FAILED ' + str(e)[:300]
                print(fam, name, i, r, flush=True)
