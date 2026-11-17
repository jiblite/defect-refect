# The moving-tone engine (phaser, filter sweep) in C++ (Source/Core/Moving.h, through drmoving) against the Python prototype
# it was ported from (Research/moving/moving.py): the same answer, the same curve, the same sounds?
#   python3 Tools/moving_parity.py path/to/drmoving model.pt a.wav [b.wav ...]      every sound: find, take off, put onto the next sound in the list
#   python3 Tools/moving_parity.py path/to/drmoving model.pt --list file.txt         the same, one path per line
# model.pt is the trainer's checkpoint; it is written out for the C++ side with Tools/export_moving_model.py first.
# Prints one line per sound and a summary; exit code 1 if an answer (found, kind, stretches) differs.
import sys, os, json, subprocess, tempfile, numpy as np, soundfile as sf, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'moving'), HERE]
import moving, pic, export_moving_model

def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def rel_db(x, ref): return 10 * np.log10(max(float(((x - ref) ** 2).sum()), 1e-300) / max(float((ref ** 2).sum()), 1e-300))

def main():
    exe, ckpt = sys.argv[1], sys.argv[2]; paths = sys.argv[3:]
    if paths and paths[0] == '--list': paths = [l.strip() for l in open(paths[1]) if l.strip()]
    tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); binp = os.path.join(tmp, 'model.bin')
    ck = torch.load(ckpt, map_location='cpu', weights_only=False); export_moving_model.export(dict(cfg=ck['cfg'], state={k: v.numpy() for k, v in ck['state'].items()}), binp); model = moving.model_of(ckpt)
    agree = 0; nfound = 0; worst_g = 0.0; worst_on = 0.0; worst_snd = -999.0; worst_sure = 0.0
    for i, p in enumerate(paths):
        pb = paths[(i + 1) % len(paths)]; a = rd(p); b = rd(pb); pre = os.path.join(tmp, 'o')
        r = subprocess.run([exe, binp, p, pb, pre], capture_output=True, text=True, check=True)
        C = json.loads([l for l in r.stdout.split('\n') if l.startswith('{')][0]); tm = ' '.join(l.strip() for l in r.stdout.split('\n') if l.startswith('  '))
        F = moving.find(model, a); T = F['T']; same = C['found'] == F['found'] and C['kind'] == F['kind'] and [tuple(s) for s in C['spans']] == [tuple(s) for s in F['spans']]; agree += int(same)
        g = np.fromfile(pre + '.g.f64').reshape(2, T, pic.NB); on = np.fromfile(pre + '.on.f64'); dg = float(np.abs(g - F['raw']).max()); do = float(np.abs(on - F['on']).max()); worst_g = max(worst_g, dg); worst_on = max(worst_on, do)
        s = '%-40s found: C++ %d %s Python %d %s%s | curve differs by at most %.1e dB, say-so by %.1e' % (os.path.basename(p)[:40], C['found'], C['kind'], F['found'], F['kind'], '' if same else '  DIFFERENT (stretches %s and %s)' % (C['spans'], F['spans']), dg, do)
        if C['found'] and F['found'] and same:
            nfound += 1; ds = abs(C['sure'] - F['sure']); dp = abs(C['period'] - F['period']); worst_sure = max(worst_sure, ds, dp); snd = {}
            for tag, ref in (('out', moving.apply(a, F, -1.0)), ('half', moving.apply(a, F, -0.5)), ('onto', moving.apply(b, F, 1.0, own=False)), ('onto2', moving.apply(b, F, 2.0, own=False)), ('fast', moving.apply(b, F, 1.0, speed=1.5, own=False)), ('slow', moving.apply(a, F, 0.0, speed=0.5, own=True))):
                snd[tag] = rel_db(np.fromfile(pre + '.%s.f64' % tag).reshape(2, -1).T, ref)
            worst_snd = max(worst_snd, max(snd.values())); s += ' | sureness %.1e, period %.1e frames | sounds %.0f dB (%s)' % (ds, dp, max(snd.values()), max(snd, key=snd.get))
        print(s + '   [C++ ' + tm + ']', flush=True)
        for f in os.listdir(tmp):
            if f != 'model.bin': os.remove(os.path.join(tmp, f))
    print('%d sounds, %d with a moving filter, %d differences in the answers. Largest difference: curve %.1e dB, say-so %.1e, sureness and period %.1e, sounds %.0f dB against the sound'
          % (len(paths), nfound, len(paths) - agree, worst_g, worst_on, worst_sure, worst_snd))
    sys.exit(0 if agree == len(paths) else 1)
main()
