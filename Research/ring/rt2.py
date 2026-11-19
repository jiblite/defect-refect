# The ring-mod prototype on sounds without the effect and on every scoreboard sound: what it claims.
#   python3 rt2.py clean i n   -> rt2_clean_<i>.jsonl        python3 rt2.py fams i n   -> rt2_fams_<i>.jsonl
import sys, json, time, os, glob; sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import ring as E
i0, n0 = int(sys.argv[2]), int(sys.argv[3])
if sys.argv[1] == 'clean':
    meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); items = [(nm, '/tmp/claude-0/v2/corpus/dry/' + nm + '.wav') for nm in sorted(meta)]
else:
    items = [(os.path.basename(os.path.dirname(os.path.dirname(d))) + '/' + os.path.basename(os.path.dirname(d)) + ':' + w, d + w + '_wet.flac') for d in sorted(glob.glob(SETS + '*/*/')) if '/reverb/' not in d for w in 'AB']
out = open('rt2_%s_%d.jsonl' % (sys.argv[1], i0), 'w')
for nm, path in items[i0::n0]:
    x = truth.rd(path); t0 = time.time(); f = E.find(x); row = dict(name=nm, secs=time.time() - t0); row.update({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in f.items() if k not in ('x', 'g')}); row['found'] = bool(f['found'])
    out.write(json.dumps(row) + '\n'); out.flush()
