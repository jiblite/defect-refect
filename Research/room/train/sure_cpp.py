# The app's own verdicts (C++ chain: both detectors + the room engine + the sureness rule) on scoreboard sounds with and
# without reverb, for cases the rule was NOT fitted on (odd places; sure_stats.py used the even ones).
#   python3 sure_cpp.py [step=4] [out=sure_cpp.json]      needs /tmp/claude-0/drbuild/drrender and the repo's Resources
import sys, os, json, glob, subprocess, time
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, soundfile as sf, truth
step = int(sys.argv[1]) if len(sys.argv) > 1 else 4; outp = sys.argv[2] if len(sys.argv) > 2 else '/tmp/claude-0/v2/train/sure_cpp.json'
EXE = '/tmp/claude-0/drbuild/drrender'; RES = '/home/claude/defect-refect/Resources'; TMP = '/tmp/claude-0/v2/train/_sure_cpp.wav'
def run(x):
    sf.write(TMP, x.astype(np.float32), 48000, subtype='FLOAT')
    o = subprocess.run(['nice', '-n', '5', EXE, RES, TMP, '-', '-'], capture_output=True, text=True).stdout.strip().split('\n')
    r = {}
    for line in o:
        if line.startswith('room: found='): r.update({'room_' + k: float(v) for k, v in (t.split('=') for t in line[5:].split())})
        elif line.startswith('room:'): r['blind'] = 1
        elif '=' in line: r['prob'] = {k: float(v) for k, v in (t.split('=') for t in line.split())}
    return r
rows = []; t0 = time.time()
ds = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[1::step] + sorted(glob.glob('/tmp/claude-0/v2/bench/sets/tone/*/'))[1::step]
for i, d in enumerate(ds):
    c = truth.load(d, 'A'); m = c['meta']; tone = '/tone/' in d
    if tone: rows.append(dict(name=c['name'], wet=0, what='tone only', **run(c['wet'])))
    else:
        direct, rev = truth.parts(c); drr = float(10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20)))
        rows.append(dict(name=c['name'], wet=1, what='reverb', drr=drr, **run(c['wet']))); rows.append(dict(name=c['name'] + ' dry', wet=0, what='dry', **run(c['dry'])))
    json.dump(rows, open(outp, 'w')); print('%d/%d %s %.0f s' % (i + 1, len(ds), c['name'], time.time() - t0), flush=True)
os.remove(TMP)
