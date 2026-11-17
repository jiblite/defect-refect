# What the app's echo engine reported on whole sounds (statscpp.py): per family, how many sounds get a DELAY group at the
# slider's default (sure > 0.65), lower and higher.
import json, glob, collections
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
S = [json.loads(l) for f in sorted(glob.glob('statscpp_*.jsonl')) for l in open(f)]; S = [s for s in S if 'error' not in s]
tags = collections.Counter(s['tag'] for s in S); took = [s['took'] for s in S]
print('%d sounds; analysis time median %.2f s, 90 %% under %.2f s, longest %.2f s' % (len(S), sorted(took)[len(took) // 2], sorted(took)[int(len(took) * 0.9)], max(took)))
print('%-12s %5s | shown at slider 50 %% | 65 %% | 80 %% | found at all (p >= 0.5)' % ('family', 'n'))
for tag in sorted(tags):
    ss = [s for s in S if s['tag'] == tag]; anyf = [s for s in ss if any(c['found'] for c in s['ch'])]
    row = '%-12s %5d |' % (tag, len(ss))
    for cut in (0.5, 0.65, 0.8): row += '  %3d (%3.0f%%) |' % (sum(s['sure'] > cut for s in anyf), 100.0 * sum(s['sure'] > cut for s in anyf) / len(ss))
    row += '  %3d' % len(anyf)
    names = ['%s %.0fms p%.2f' % (s['name'] if tag != 'dry' or meta[s['name']]['kind'] != 'real-recording' else 'rec', max(c['found']['q'] for c in s['ch'] if c['found']) / 48.0, s['p']) for s in anyf if s['sure'] > 0.65]
    print(row + ('   ' + ' | '.join(names[:12]) if tag not in ('delay',) and names else ''))
