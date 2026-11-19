# Does build 0.5.0 leave everything else alone? One case of every effect type through the render test of build 0.4.7 and through the new one,
# with every fader in the middle: the first sound alone and onto the second sound. The results must be the same to the last bit.
#   python3 same.py [0|1 ...]     which case of each type's bench set (0 = the first); OLD and NEW are the two builds of Tests/render.cpp
import sys, os, glob, subprocess, numpy as np, soundfile as sf
sys.path[:0] = ['/tmp/claude-0/v2/sweep']; from lab import load
OLD, NEW = '/tmp/claude-0/drrender047g', '/tmp/claude-0/v2/mv/cc/drrender_v9'; RES = '/home/claude/defect-refect/Resources'; SETS = '/tmp/claude-0/v2/bench/sets/'; D = '/dev/shm/same/'; os.makedirs(D, exist_ok=True)
which = int(sys.argv[1]) if len(sys.argv) > 1 else 0; bad = 0
for fam in sorted(os.listdir(SETS)):
    cases = sorted(glob.glob(SETS + fam + '/*/'))
    if len(cases) <= which: continue
    c = cases[which]; a = load(c, 'A')['wet']; b = load(c, 'B')['dry']; sf.write(D + 'a.wav', a, 48000, subtype='FLOAT'); sf.write(D + 'b.wav', b, 48000, subtype='FLOAT'); res = []
    for exe, tag in ((OLD, 'o'), (NEW, 'n')):
        r = subprocess.run([exe, RES, D + 'a.wav', '-', D + tag + '1.wav', '--', D + 'b.wav', D + tag + '2.wav'], capture_output=True, text=True)
        if r.returncode: print(fam, tag, 'FAILED', r.stderr[-200:]); res = None; break
        res.append([sf.read(D + tag + '%d.wav' % k, dtype='float32', always_2d=True)[0] for k in (1, 2)])
    if res is None: bad += 1; continue
    same = [res[0][k].shape == res[1][k].shape and np.array_equal(res[0][k], res[1][k]) for k in (0, 1)]
    diff = [float(np.abs(res[0][k] - res[1][k]).max()) if res[0][k].shape == res[1][k].shape else -1.0 for k in (0, 1)]
    print('%-12s %-28s alone: %s  onto the second: %s' % (fam, os.path.basename(c[:-1]), 'same' if same[0] else 'DIFFERS by %.2e' % diff[0], 'same' if same[1] else 'DIFFERS by %.2e' % diff[1]), flush=True); bad += not all(same)
print('cases that differ:', bad)
