# Does the level engine report a movement on sounds that carry ANOTHER effect (scoreboard families other than its own), on the
# reverb set and on the tone set?   python3 others.py PART NPARTS   -> others_<PART>.jsonl
import sys, os, json, time, numpy as np
sys.path += ['/tmp/claude-0/v2/proto', '/tmp/claude-0/v2/bench', '/tmp/claude-0/v2/level']
import truth, level
S = '/tmp/claude-0/v2/bench/sets/'; part, nparts = int(sys.argv[1]), int(sys.argv[2]); out = open('others_%d.jsonl' % part, 'w'); jobs = []
for fam in sorted(os.listdir(S)):
    names = sorted(os.listdir(S + fam))
    if fam in ('reverb', 'tone'): names = names[::3]
    for name in names: jobs.append((fam, name))
for i, (fam, name) in enumerate(jobs):
    if i % nparts != part: continue
    for tag in ('A', 'B'):
        try: c = truth.load(S + fam + '/' + name + '/', tag)
        except Exception as e: continue
        for what in ('wet',) + (('dry',) if fam in ('reverb', 'tone') else ()):
            t0 = time.time(); F = level.find(c[what]); row = dict(fam=fam, name=name, dir=tag, what=what, found=F is not None, secs=time.time() - t0, n=len(c[what]))
            if F is not None: row.update(kind=F['kind'], rate=F['rate'], depth=F['depth'])
            out.write(json.dumps(row) + '\n'); out.flush()
