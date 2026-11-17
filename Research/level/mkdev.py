# More clean sounds for the rule: every candidate find() sees on the dry corpus sounds of the 'dev' split (never 'test').
#   python3 mkdev.py   -> data/corpus_dev.jsonl     (ids and measured numbers only)
import sys, os, json, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); out = open('/tmp/claude-0/v2/level/data/corpus_dev.jsonl', 'w')
def strip(m): return {k: (v if not isinstance(v, (np.floating, np.integer, np.bool_)) else float(v)) for k, v in m.items() if k not in ('g', 'prof', 'lane')} if m else None
for cid in sorted(meta):
    if meta[cid]['split'] != 'dev': continue
    x, sr = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % cid, dtype='float64', always_2d=True); x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1); E = level.levels(x)
    out.write(json.dumps(dict(name=cid, kind=meta[cid]['kind'], pan=strip(level.find_pan(E)), tog=[strip(m) for m in level.candidates_together(E)])) + '\n'); out.flush()
