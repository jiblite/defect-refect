# What the analysis costs on an idle machine: 8 clean long sounds and 4 corpus sounds of 6 s, each tool alone, one after the other.
#   python3 timing.py path/drlevel path/drsweep path/drrender Resources
import sys, os, re, json, subprocess, tempfile, numpy as np, soundfile as sf
import long as L
lv, sw, rd, res = sys.argv[1:5]; rows = []
def app(p, d):
    o = subprocess.run([rd, res, p, '-', os.path.join(d, 'out.wav')], capture_output=True, text=True).stdout; m = re.search(r'inspect (\d+) ms', o)
    return int(m.group(1)) / 1000.0 if m else float('nan')
for i in range(8):
    seconds = L.LENGTHS[i % 4]; rng = np.random.default_rng(7000 + i); x, kind, used = L.long_sound(rng, seconds, i < 4)
    d = tempfile.mkdtemp(dir='/dev/shm'); p = os.path.join(d, 'a.wav'); sf.write(p, x, L.SR, subtype='FLOAT')
    a = json.loads(subprocess.run([lv, 'pieces', p], capture_output=True, text=True).stdout)['took']
    b = json.loads(subprocess.run([sw, 'both', p, p, os.path.join(d, 'o'), os.path.join(d, 'q')], capture_output=True, text=True).stdout)['took']
    c = app(p, d)
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    os.rmdir(d); rows.append((seconds, a, b, c)); print('%2d s %-14s level %6.2f s (%.3f per s) | flanger %6.2f s (%.3f per s) | the app, everything %6.2f s (%.3f per s)' % (seconds, kind, a, a / seconds, b, b / seconds, c, c / seconds), flush=True)
print('long sounds, per second of sound: level %.3f s, flanger %.3f s, the app (all engines) %.3f s' % tuple(np.mean([r[k] / r[0] for r in rows]) for k in (1, 2, 3)))
names = sorted(L.META)[::max(1, len(L.META) // 4)][:4]; six = []
for nm in names:
    d = tempfile.mkdtemp(dir='/dev/shm'); p = os.path.join(d, 'a.wav'); x = L.clip(nm); sf.write(p, x, L.SR, subtype='FLOAT'); s = len(x) / L.SR
    a = json.loads(subprocess.run([lv, 'pieces', p], capture_output=True, text=True).stdout)['took']
    b = json.loads(subprocess.run([sw, 'both', p, p, os.path.join(d, 'o'), os.path.join(d, 'q')], capture_output=True, text=True).stdout)['took']
    c = app(p, d)
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    os.rmdir(d); six.append((s, a, b, c)); print('%.1f s corpus sound: level %.2f s | flanger %.2f s | the app, everything %.2f s' % (s, a, b, c), flush=True)
print('sounds of 6 s: level %.2f s, flanger %.2f s, the app (all engines) %.2f s (means)' % tuple(np.mean([r[k] for r in six]) for k in (1, 2, 3)))
