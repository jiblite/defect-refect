# The fitted rule on whole sounds (stats2.py): per sound the surest lag of each channel.
#   python3 judge3.py rule3.pkl [stats files ...]
import sys, json, glob, pickle, collections, numpy as np
sys.path.insert(0, '.'); import echo2
m = pickle.load(open(sys.argv[1], 'rb')); files = sys.argv[2:] or sorted(glob.glob('stats2_*.jsonl'))
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
S = [json.loads(l) for f in files for l in open(f)]
for s in S:
    for c in s['ch']:
        if c['recs']:
            P = m.predict_proba(np.array([echo2.feats(r) for r in c['recs']], float))[:, 1]
            for r, p in zip(c['recs'], P): r['p'] = float(p)
def best(c): return max(c['recs'], key=lambda r: r['p']) if c['recs'] else None
tags = collections.Counter(s['tag'] for s in S); print(dict(tags))
for cut in (0.5, 0.65, 0.775, 0.85, 0.928, 0.962):
    line = 'p >= %.3f:' % cut
    for tag in tags:
        if tag == 'delay':
            ok = wrong = none = snd = 0
            for s in [s for s in S if s['tag'] == tag]:
                any_ok = False
                for ch, c in enumerate(s['ch']):
                    b = best(c)
                    if b is None or b['p'] < cut: none += 1
                    else:
                        r = b['q'] / s['true'][ch]; good = abs(r - 1) < 0.004 or (round(r) >= 2 and abs(r - round(r)) < 0.004) or (round(1 / r) >= 2 and abs(1 / r - round(1 / r)) < 0.004)
                        ok += good; wrong += not good; any_ok |= good
                snd += any_ok
            line += '  delay: channels right %d wrong %d none %d, sounds found %d of %d |' % (ok, wrong, none, snd, tags[tag])
        else:
            rep = [s for s in S if s['tag'] == tag and any(best(c) is not None and best(c)['p'] >= cut for c in s['ch'])]
            line += '  %s: reported %d of %d' % (tag, len(rep), tags[tag])
            if tag == 'dry':
                kinds = collections.Counter(meta[s['name']]['kind'] for s in rep); line += ' %s' % dict(kinds)
                if cut in (0.65, 0.775, 0.928):
                    line += '\n      ' + ' | '.join('%s %.0fms a%.2f p%.2f' % (s['name'] if meta[s['name']]['kind'] != 'real-recording' else 'rec', max((best(c) for c in s['ch'] if best(c)), key=lambda r: r['p'])['ms'], max((best(c) for c in s['ch'] if best(c)), key=lambda r: r['p'])['a'], max((best(c) for c in s['ch'] if best(c)), key=lambda r: r['p'])['p']) for s in rep[:30])
            elif cut in (0.775,):
                line += '\n      ' + ' | '.join('%s %.0fms a%.2f p%.2f' % (s['name'], max((best(c) for c in s['ch'] if best(c)), key=lambda r: r['p'])['ms'], max((best(c) for c in s['ch'] if best(c)), key=lambda r: r['p'])['a'], max((best(c) for c in s['ch'] if best(c)), key=lambda r: r['p'])['p']) for s in rep[:30])
    print(line)
if 'delay' in tags:
    print('\ndelay sounds, per channel: true lag ms | best record (ms, a, p) | all records')
    for s in [s for s in S if s['tag'] == 'delay']:
        print('  %-20s %s' % (s['name'], ' || '.join('true %.1f: %s' % (s['true'][ch] / 48.0, ' '.join('%.1fms a%.2f p%.3f' % (r['ms'], r['a'], r['p']) for r in sorted(c['recs'], key=lambda r: -r['p'])) or 'nothing measured (%s)' % ' '.join('%.1f/%.2f%s' % (k['q'] / 48.0, k['a'], 'r' if k.get('rah') else '') for k in c['cand'][:4])) for ch, c in enumerate(s['ch']))))
