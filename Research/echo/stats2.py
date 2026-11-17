# What echo2.measure reports on whole sounds: the scoreboard's delay sounds (truth known), the 198 clean corpus sounds, and (asked for
# by name) the wet sounds of other effect families. One line per sound and channel: every measured record.
#   python3 stats2.py part of [family ...]     -> stats2_<part>.jsonl   (no family: delay + the clean corpus)
import sys, os, json, glob, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '.')
import truth, echo2 as echo
part, of = int(sys.argv[1]), int(sys.argv[2]); fams = sys.argv[3:]; fn = 'stats2_%d.jsonl' % part
done = {(json.loads(l)['tag'], json.loads(l)['name']) for l in open(fn)} if os.path.exists(fn) else set(); out = open(fn, 'a')
jobs = []; S = '/tmp/claude-0/v2/bench/sets/'
if not fams:
    for name in sorted(os.listdir(S + 'delay')):
        for w in 'AB': jobs.append(('delay', name + '_' + w, (S + 'delay/' + name + '/', w)))
    for f in sorted(glob.glob('/tmp/claude-0/v2/corpus/dry/*.wav')): jobs.append(('dry', os.path.basename(f)[:-4], f))
for fam in fams:
    for name in sorted(os.listdir(S + fam)):
        for w in 'AB': jobs.append((fam, name + '_' + w, S + fam + '/' + name + '/' + w + '_wet.flac'))
for tag, name, src in jobs[part::of]:
    if (tag, name) in done: continue
    true = None
    if tag == 'delay':
        c = truth.load(src[0], src[1]); x = c['wet']; true = []
        for ch in range(2):
            h = np.abs(c['h'][ch][:, ch]); h[:480] = 0; true.append(int(np.argmax(h)))
    else:
        x, sr = sf.read(src, dtype='float64', always_2d=True); x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
    s, n = echo.stretch(x); chans = []
    for ch in range(2):
        cand, recs = echo.measure(x[s:s + n, ch]); chans.append(dict(cand=cand[:8], recs=[r for r, _ in recs]))
    out.write(json.dumps(dict(tag=tag, name=name, true=true, secs=len(x) / 48000.0, start=s, ch=chans)) + '\n'); out.flush()
