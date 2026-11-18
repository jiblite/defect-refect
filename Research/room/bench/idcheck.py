# The render tool of two builds on scoreboard sounds (6 and 9 s): must give the same text and the same output file.
import sys, glob, os, subprocess, hashlib, re, soundfile as sf
OLD, NEW = sys.argv[1], sys.argv[2]; step = int(sys.argv[3]); off = int(sys.argv[4]); RES = '/home/claude/defect-refect/Resources'
cases = sorted(glob.glob('/tmp/claude-0/v2/bench/sets/*/*/'))[off::step]; same = diff = 0
def run(exe, pin, pout):
    q = subprocess.run([exe, RES, pin, '-', pout, 'reverb.amount=0'], capture_output=True, text=True)
    t = '\n'.join(l for l in q.stdout.splitlines() if not re.search(r'\bms\b|seconds|took|time', l)); return q.returncode, t, hashlib.md5(open(pout, 'rb').read()).hexdigest() if os.path.exists(pout) else None
for d in cases:
    for w in ('A', 'B'):
        y, sr = sf.read(d + w + '_wet.flac', dtype='float64', always_2d=True); pin = 'in_%d.wav' % off; sf.write(pin, y, sr, subtype='FLOAT')
        a = run(OLD, pin, 'o_old_%d.wav' % off); b = run(NEW, pin, 'o_new_%d.wav' % off); ok = a == b; same += ok; diff += not ok
        m = re.search(r'^room: found=(\d) .*? sureness=([0-9.eE+-]+)', b[1], re.M); m2 = re.search(r'room without the other effects.*', b[1])
        print('%-9s %-28s %s %4.1f s  room %s %s' % ('same' if ok else 'DIFFERENT', d.split('/')[-2][:28], w, len(y) / sr, m.group(2) if m else '-', ('| ' + m2.group(0)[:70]) if m2 else ''), flush=True)
        if not ok:
            la, lb = a[1].splitlines(), b[1].splitlines()
            for x, z in zip(la, lb):
                if x != z: print('   old:', x[:200]); print('   new:', z[:200])
print('same %d, different %d' % (same, diff))
