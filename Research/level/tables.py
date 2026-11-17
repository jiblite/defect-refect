# The write-ups' numbers for long sounds from the final files: python3 tables.py   (reads long_<set>_p7, long_<set>_s2, fast_third_k.jsonl)
import os, json, numpy as np
import long as L, lsum
def r0(v): return '%.0f' % (round(v) + 0.0)
def rows(mode, tag):
    p = 'long_%s_%s.jsonl' % (mode, tag); return sorted((json.loads(l) for l in open(p)), key=lambda r: r['i']) if os.path.exists(p) else []
NAMES = (('tremolo', 'tremolo'), ('autopan', 'auto pan'), ('gatedrhythm', 'gate'))
def table(tag, sets):
    S = lsum.summary(tag); out = []
    if 'clean' in S: out.append('| clean                        | %-6d | %-5d |                                |                     |                    |' % (S['clean']['n'], S['clean']['shown']))
    for mode, label in sets:
        if mode not in S: continue
        for fam, name in NAMES:
            if fam not in S[mode]: continue
            f = S[mode][fam]; worse = []
            rs = [r for r in rows(mode, tag) if r['fam'] == fam]
            wo = [r for r in rs if L.right(r) and r['out'] < 0]; wn = [r for r in rs if L.right(r) and r['onto'] < 0]
            w = '0' if not wo and not wn else ', '.join(([('%d (%s)' % (len(wo), ', '.join(r0(r['out']) for r in wo)))] if wo else []) + ([('%d onto (%s)' % (len(wn), ', '.join(r0(r['onto']) for r in wn)))] if wn else []))
            out.append('| %-28s | %-6d | %-5d | %-30s | %-19s | %-18s |' % (label + ': ' + name, f['n'], f['shown'], '%s, %s' % (r0(f['out_med']), r0(f['out_shown'])), '%s, %s' % (r0(f['onto_med']), r0(f['onto_shown'])), w))
    return S, out
def blocks(tag):
    v = {}
    for fam, name in NAMES:
        bl = [b for r in rows('wet', tag) if r['fam'] == fam and L.right(r) for b in r.get('out_blocks', [])]; v[name] = 100 * float(np.mean(np.array(bl) < 50)) if bl else float('nan')
    return v
import sys
T1, T2 = (sys.argv[1:3] + ['p7', 's2'])[:2] if len(sys.argv) > 2 else ('p7', 's2')
for tag, title in ((T1, 'FIRST DRAW'), (T2, 'SECOND DRAW')):
    S, t = table(tag, (('wet', 'whole sound'), ('part', 'one half'), ('drift', 'wandering speed'), ('slow', 'slow pans (3 to 23 s)')))
    print('==', title, tag); print('\n'.join(t))
    if 'slow' in S: f = S['slow']['all']; print('| slow pans (3 to 23 s)        | %-6d | %-5d | %s, %s | %s, %s | %d |' % (f['n'], f['shown'], r0(f['out_med']), r0(f['out_shown']), r0(f['onto_med']), r0(f['onto_shown']), f['out_worse']))
    for mode in ('wet', 'part', 'drift'):
        if mode in S: f = S[mode]['all']; print('   %-5s all: shown %d of %d (lower Sureness %d, other kind %d), out median %.1f (where shown %.1f), onto %.1f (%.1f), worse out %d onto %d' % (mode, f['shown'], f['n'], f['lower'], f['wrong'], f['out_med'], f['out_shown'], f['onto_med'], f['onto_shown'], f['out_worse'], f['onto_worse']))
    if 'clean' in S: print('   clean: shown %d, at the lowest Sureness %d of %d' % (S['clean']['shown'], S['clean']['lowest'], S['clean']['n']))
    if 'part' in S and 'off' in S['part']: o = S['part']['off']; print('   the half without the effect: unchanged (under -150 dB) in %d of %d shown; the rest at %s dB' % (o['exact'], o['n'], o['rest']))
    b = blocks(tag); print('   6 s stretches under 50, of the sounds shown: ' + ', '.join('%s %.0f %%' % (k, v) for k, v in b.items()))
    for mode in ('wet', 'part', 'drift'):
        for r in rows(mode, tag):
            if L.right(r) and (r['out'] < 0 or r['onto'] < 0): print('   worse than nothing: %s #%d %s %d s out %.1f onto %.1f' % (mode, r['i'], r['effect'], r['seconds'], r['out'], r['onto']))
if os.path.exists('fast_third_k.jsonl'):
    R = [json.loads(l) for l in open('fast_third_k.jsonl')]; print('== THIRD DRAW (whole-sound scores; i = before the gate fixes, k = final)')
    def any_shown(v): return v.get('found') and v.get('score') is not None and v['score'] >= 0
    def shown(v, fam=None): return any_shown(v) and (fam is None or v['kind'] == L.WANT[fam])
    for mode, label in (('wet', 'whole sound'), ('part', 'one half'), ('drift', 'wandering speed')):
        for fam, name in NAMES:
            rs = [r for r in R if r['mode'] == mode and r['fam'] == fam]
            if not rs: continue
            line = '   %-16s %-9s n %2d' % (label, name, len(rs))
            for t in ('i', 'k'):
                o = np.array([r['v'][t]['out'] if shown(r['v'][t], fam) else 0.0 for r in rs]); n = np.array([r['v'][t]['onto'] if shown(r['v'][t], fam) else 0.0 for r in rs]); sh = [r for r in rs if shown(r['v'][t], fam)]
                other = [r for r in rs if any_shown(r['v'][t]) and not shown(r['v'][t], fam)]
                line += ' | %s shown %2d (other kind %d) out %5.1f (where shown %5.1f) onto %5.1f (%5.1f) worse %d/%d' % (t, len(sh), len(other), np.median(o), np.median([r['v'][t]['out'] for r in sh]) if sh else 0, np.median(n), np.median([r['v'][t]['onto'] for r in sh]) if sh else 0, int((o < 0).sum()), int((n < 0).sum()))
                if mode == 'part': off = [r['v'][t].get('out_off', -999) for r in sh]; line += ' clean half: unchanged %d of %d, rest %s' % (sum(v <= -150 for v in off), len(off), sorted(round(v) for v in off if v > -150))
            print(line)
    for r in R:
        v = r['v']['k']
        if any_shown(v) and (v['out'] < 0 or v['onto'] < 0): print('   worse than nothing (final): %s #%d %s %d s kind %s out %.1f onto %.1f | before: kind %s out %.1f onto %.1f' % (r['mode'], r['i'], r['effect'], r['seconds'], v['kind'], v['out'], v['onto'], r['v']['i']['kind'], r['v']['i']['out'], r['v']['i']['onto']))
