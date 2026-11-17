# What the app's echo engine (drecho measure) reports on whole sounds: the clean corpus and the wet sounds of every effect family
# of the scoreboard. One line per sound.
#   python3 statscpp.py part of  -> statscpp_<part>.jsonl
import sys, os, json, glob, subprocess, numpy as np, soundfile as sf
part, of = int(sys.argv[1]), int(sys.argv[2]); fn = 'statscpp_%d.jsonl' % part; tool = '/tmp/claude-0/drbuild/drecho'
done = {(json.loads(l)['tag'], json.loads(l)['name']) for l in open(fn)} if os.path.exists(fn) else set(); out = open(fn, 'a')
jobs = []; S = '/tmp/claude-0/v2/bench/sets/'
for f in sorted(glob.glob('/tmp/claude-0/v2/corpus/dry/*.wav')): jobs.append(('dry', os.path.basename(f)[:-4], f))
for fam in sorted(os.listdir(S)):
    for name in sorted(os.listdir(S + fam)):
        for w in 'AB':
            if os.path.exists(S + fam + '/' + name + '/' + w + '_wet.flac'): jobs.append((fam, name + '_' + w, S + fam + '/' + name + '/' + w + '_wet.flac'))
tmp = '/dev/shm/ep/stat_%d.wav' % part
for tag, name, src in jobs[part::of]:
    if (tag, name) in done: continue
    x, sr = sf.read(src, dtype='float32', always_2d=True)
    if sr != 48000: continue
    sf.write(tmp, x[:, :2], 48000, subtype='FLOAT'); p = subprocess.run([tool, 'measure', tmp], capture_output=True, text=True)
    if p.returncode != 0: out.write(json.dumps(dict(tag=tag, name=name, error=p.stderr[-200:])) + '\n'); continue
    r = json.loads(p.stdout); r.update(tag=tag, name=name); out.write(json.dumps(r) + '\n'); out.flush()
os.path.exists(tmp) and os.remove(tmp)
