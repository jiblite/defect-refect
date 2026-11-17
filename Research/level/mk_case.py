# writes the wet and dry sound of one long test case: python3 mk_case.py mode i [SEED]  ->  /dev/shm/lc/<mode>_<i>_{wet,dry,dryb,wetb}.wav
import sys, os, numpy as np, soundfile as sf
import long as L
mode, i = sys.argv[1], int(sys.argv[2]); os.makedirs('/dev/shm/lc', exist_ok=True)
if mode == 'wet': e, j = divmod(i, 8); name = L.EFFECTS[e]; seconds = L.LENGTHS[j % 4]; same = j < 4
elif mode == 'part': e, j = divmod(i, 4); name = L.EFFECTS[e]; seconds = L.LENGTHS[1 + j % 3]; same = j < 2
elif mode == 'drift': e, j = divmod(i, 8); name = ('trem', 'pan')[e]; seconds = L.LENGTHS[j % 4]; same = j < 4
elif mode == 'slow': e, j = divmod(i, 8); name = ('pan_sox', 'pan_tap', 'pan_calf')[e]; seconds = L.LENGTHS[j % 4]; same = j < 4
rng = np.random.default_rng(L.SEED + dict(wet=9000, part=11000, drift=13000, slow=15000)[mode] + i); n = int(seconds * L.SR)
fn, fam, par, live = L.effect_of(mode, name, rng, n); dry, kind, used = L.long_sound(rng, seconds, same); dryb, kindb, usedb = L.long_sound(rng, seconds, same)
for tag, x in (('dry', dry), ('wet', fn(dry)), ('dryb', dryb), ('wetb', fn(dryb))): sf.write('/dev/shm/lc/%s_%d_%s.wav' % (mode, i, tag), x, L.SR, subtype='FLOAT')
print(name, fam, par, seconds, kind, ' '.join(used))
