# Does silence after the sound change the REVERB verdict?  E1: dry corpus sounds as files and with 3 s of silence after them.
# E2: the reverb scoreboard's wet sounds (sound A): whole (with the tail ringing out), cut at 6 s (the tail cut off), cut at 6 s + 3 s of silence;
#     and their dry versions the same three ways.
#   python3 cut_test.py dry PART NPARTS | python3 cut_test.py rev PART NPARTS     -> padtest_<what>_<part>.jsonl (first look: [found, sureness])
# Run with a drrender built before the second look went in (Estimate.h, recheckRoomEnd), so that "sureness" is the first look's.
import sys, os, json, glob, subprocess, re, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/proto'); import truth
EXE = '/tmp/claude-0/v2/bench/drrender_042'; RES = '/home/claude/defect-refect/Resources'; what, part, of = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); TMP = '/dev/shm/ep/rc/p%s%d.wav' % (what, part)
out = '/tmp/claude-0/v2/bench/padtest_%s_%d.jsonl' % (what, part); done = {json.loads(l)['name'] for l in open(out)} if os.path.exists(out) else set()
def verdict(x):
    sf.write(TMP, x.astype(np.float32), 48000, subtype='FLOAT'); o = subprocess.run([EXE, RES, TMP, '-', '-'], capture_output=True, text=True).stdout
    m = re.search(r'room: found=(\d) detector=\S+ sureness=([0-9.eE+-]+)', o); return [int(m.group(1)), float(m.group(2))] if m else None
Z = lambda x, s=3.0: np.concatenate([x, np.zeros((int(s * 48000), x.shape[1]))])
if what == 'dry':
    for f in sorted(glob.glob('/tmp/claude-0/v2/corpus/dry/*.wav'))[part::of]:
        n = os.path.basename(f)[:-4]
        if n in done: continue
        x, sr = sf.read(f, dtype='float64', always_2d=True); open(out, 'a').write(json.dumps(dict(name=n, asis=verdict(x), pad=verdict(Z(x)), pad1=verdict(Z(x, 1.0)))) + '\n')
else:
    for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*/'))[part::of]:
        c = truth.load(d, 'A')
        if c['name'] in done: continue
        w, dr = c['wet'], c['dry']; n6 = 6 * 48000
        open(out, 'a').write(json.dumps(dict(name=c['name'], n=len(w), wet=verdict(w), wet_cut=verdict(w[:n6]), wet_cut_pad=verdict(Z(w[:n6])), dry=verdict(dr), dry_cut=verdict(dr[:n6]), dry_cut_pad=verdict(Z(dr[:n6])))) + '\n')
