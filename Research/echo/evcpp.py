# The app's echo engine (drecho) scored on the scoreboard's 8 delay cases, both ways: echoes out of the sound they are on, and onto the
# case's other sound. 100 = identical to the truth, 0 = like doing nothing.
import sys, os, json, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto')
import truth, score
tool = '/tmp/claude-0/drbuild/drecho'; S = '/tmp/claude-0/v2/bench/sets/delay/'; D = '/dev/shm/ep/'; tot = []; shown = []; extra = sys.argv[1:]
for name in sorted(os.listdir(S)):
    c = {w: truth.load(S + name + '/', w) for w in 'AB'}; res = []
    for w, o in (('A', 'B'), ('B', 'A')):
        a = D + '%s_%s_wet.wav' % (name, w); b = D + '%s_%s_dry.wav' % (name, o)
        p = subprocess.run([tool, 'out', a, D + 'o.wav'] + extra, capture_output=True, text=True); info = p.stderr.strip()
        sure = float(info.split('(sure ')[1].split(')')[0]) if '(sure ' in info else 0.0
        if 'found 0 0' in info: res.append('not found'); tot.append((0.0, 0.0)); shown.append((0.0, 0.0)); continue
        y, _ = sf.read(D + 'o.wav', dtype='float64'); subprocess.run([tool, 'onto', a, b, D + 'p.wav'], capture_output=True, text=True); z, _ = sf.read(D + 'p.wav', dtype='float64')
        r = score.removed(c[w]['dry'], c[w]['wet'], y)[0]; ro = score.removed(c[o]['wet'], c[o]['dry'], z)[0]; tot.append((r, ro)); shown.append((r, ro) if sure > 0.65 else (0.0, 0.0)); res.append('out %5.1f onto %5.1f  (%s)%s' % (r, ro, info, '' if sure > 0.65 else '  NOT SHOWN at the default Sureness'))
    print('%-18s A: %s\n%-18s B: %s' % (name, res[0], '', res[1]), flush=True)
t = np.array(shown); print('as the app shows it at the default Sureness (65 %%; a sound whose echo is not shown counts as 0): %d of %d shown | out: median %.1f mean %.1f | onto: median %.1f mean %.1f' % ((t[:, 0] != 0).sum(), len(t), np.median(t[:, 0]), t[:, 0].mean(), np.median(t[:, 1]), t[:, 1].mean()))
t = np.array(tot); print('every sound where the engine has an echo at all (p >= 0.5): ', end='')
print('out: median %.1f mean %.1f | onto: median %.1f mean %.1f | n=%d' % (np.median(t[:, 0]), t[:, 0].mean(), np.median(t[:, 1]), t[:, 1].mean(), len(t)))
