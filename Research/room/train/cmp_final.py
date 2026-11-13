# Side by side: two models on the same final_eval cases.   python3 cmp_final.py modelA modelB
import sys, os, json, numpy as np
T = '/tmp/claude-0/v2/train/'; A, B = sys.argv[1], sys.argv[2]; ra = {r['name']: r for r in json.load(open(T + 'final_eval_%s.json' % A))}; rb = {r['name']: r for r in json.load(open(T + 'final_eval_%s.json' % B))}
names = sorted(set(ra) & set(rb)); OWN = json.load(open(os.environ.get('SOURCE_WIDTH', 'source_width.json')))      # per case: how much of the dry sound left and right do not share (dB)
def st(v): v = np.array(v); return 'median %5.1f mean %5.1f below zero %2.0f%%' % (np.median(v), v.mean(), 100 * (v < 0).mean())
print('%d cases | room found: %s %.0f%%, %s %.0f%%' % (len(names), A, 100 * np.mean([ra[n]['found'] for n in names]), B, 100 * np.mean([rb[n]['found'] for n in names])))
for k in ra[names[0]]['out']:
    print('%-26s %-22s %s\n%-26s %-22s %s' % (k, A, st([ra[n]['out'][k] for n in names]), '', B, st([rb[n]['out'][k] for n in names])))
for lab, sel in (('mono or narrow sound', [n for n in names if OWN.get(n, 0) < -20]), ('wide stereo sound', [n for n in names if OWN.get(n, 0) >= -20]), ('dev', [n for n in names if ra[n]['split'] == 'dev']), ('test', [n for n in names if ra[n]['split'] == 'test']),
                 ('light DRR>=8', [n for n in names if ra[n]['drr'] >= 8]), ('middle', [n for n in names if 0 <= ra[n]['drr'] < 8]), ('wet DRR<0', [n for n in names if ra[n]['drr'] < 0])):
    if sel: print('%-22s n=%2d reverb out | %s: %s | %s: %s' % (lab, len(sel), A, st([ra[n]['out']['reverb out'] for n in sel]), B, st([rb[n]['out']['reverb out'] for n in sel])))
for k in ('tail_cut', 'body_cut', 'tail_cut_bold', 'body_cut_bold', 'rev_cut', 'dir_cut'):
    print('%-14s %s %.2f | %s %.2f (medians, dB)' % (k, A, np.nanmedian([ra[n][k] for n in names]), B, np.nanmedian([rb[n][k] for n in names])))
d = np.array([rb[n]['out']['reverb out'] - ra[n]['out']['reverb out'] for n in names]); print('per case, reverb out, %s minus %s: better on %d, worse on %d, median %+.1f; worst %s' % (B, A, (d > 1).sum(), (d < -1).sum(), np.median(d), ', '.join('%s %+.0f' % (names[i], d[i]) for i in np.argsort(d)[:4])))
