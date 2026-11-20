# Many soft clipper cases for setting the declipper's second safeguard: ffmpeg's soft clippers at their own rate (a
# curve's work) and working at 2 and 4 times the rate (flat tops with ripple: not a curve's work), and pedalboard's
# distortion. Clean clips drawn from the whole corpus. WAVs (24 bit) and a list of settings. Container only: the folder
# holds audio; no clip names are printed or stored.
#   DR_SETS=/tmp/claude-0/v2/bench/sets/ DR_CORPUS=/tmp/claude-0/v2/corpus/ python3 mk_many.py out_dir per_cell [seed]
import sys, os, json
sys.path.insert(0, '/home/claude/defect-refect/Research/detect'); sys.path.insert(0, '/home/claude/defect-refect/Research/declip')
out, per = sys.argv[1], int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5; sys.argv = sys.argv[:1]
import numpy as np, soundfile as sf
import mkpos, clip_fx, zoo_fx as Z
os.makedirs(out, exist_ok=True); rng = np.random.default_rng(seed); names = sorted(mkpos.meta); rows = []
cells = [('clip_ff_' + k, o) for k in clip_fx.KINDS if k != 'hard' for o in (1, 2, 4)] + [('dist_pedalboard', 1)]
for eff, o in cells:
    for i in range(per if eff != 'dist_pedalboard' else 3 * per):
        n = str(rng.choice(names)); x = mkpos.dry(n); p = dict(drive_db=float(rng.uniform(14, 34))) if eff == 'dist_pedalboard' else dict(drive_db=float(rng.uniform(8, 28)), oversample=o)
        w = Z.apply(eff, x, p, 0.0)[:len(x)]
        if not np.isfinite(w).all() or np.abs(w).max() < 1e-3: continue
        pk = np.abs(w).max(); g = 1.0 if pk < 0.98 else 0.98 / pk; name = 'm_%s_%d_%02d' % (eff.replace('clip_ff_', '').replace('dist_', ''), o, i)
        sf.write(os.path.join(out, name + '_wet.wav'), w * g, 48000, subtype='PCM_24'); sf.write(os.path.join(out, name + '_dry.wav'), x / max(1.0, np.abs(x).max()), 48000, subtype='PCM_24')
        rows.append(dict(name=name, effect=eff, over=o, drive=p['drive_db'], kind='own' if 'real' in n else n.split('_')[0]))
    print(eff, o, len(rows), flush=True)
json.dump(rows, open(os.path.join(out, 'list.json'), 'w'))
