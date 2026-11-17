# The app's echo engine (drecho) scored on the scoreboard's 8 delay cases, both ways: echoes out of the sound they are on, and onto the
# case's other sound. 100 = identical to the truth, 0 = like doing nothing.
import sys, os, json, subprocess, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto')
import truth, score
tool = '/tmp/claude-0/drbuild/drecho'; S = '/tmp/claude-0/v2/bench/sets/delay/'; D = '/dev/shm/ep/'; tot = []; extra = sys.argv[1:]
for name in sorted(os.listdir(S)):
    c = {w: truth.load(S + name + '/', w) for w in 'AB'}; res = []
    for w, o in (('A', 'B'), ('B', 'A')):
        a = D + '%s_%s_wet.wav' % (name, w); b = D + '%s_%s_dry.wav' % (name, o)
        p = subprocess.run([tool, 'out', a, D + 'o.wav'] + extra, capture_output=True, text=True); info = p.stderr.strip()
        if 'found 0 0' in info: res.append('not found'); tot.append((0.0, 0.0)); continue
        y, _ = sf.read(D + 'o.wav', dtype='float64'); subprocess.run([tool, 'onto', a, b, D + 'p.wav'], capture_output=True, text=True); z, _ = sf.read(D + 'p.wav', dtype='float64')
        r = score.removed(c[w]['dry'], c[w]['wet'], y)[0]; ro = score.removed(c[o]['wet'], c[o]['dry'], z)[0]; tot.append((r, ro)); res.append('out %5.1f onto %5.1f  (%s)' % (r, ro, info))
    print('%-18s A: %s\n%-18s B: %s' % (name, res[0], '', res[1]), flush=True)
t = np.array(tot); print('out: median %.1f mean %.1f | onto: median %.1f mean %.1f | n=%d' % (np.median(t[:, 0]), t[:, 0].mean(), np.median(t[:, 1]), t[:, 1].mean(), len(t)))
