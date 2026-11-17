# single cases of flong.py's sets: python3 fone.py path/to/drsweep wet|part i [i ...]
import sys, json, numpy as np
import flong as F
exe, mode = sys.argv[1], sys.argv[2]; want = [int(v) for v in sys.argv[3:]]
jobs = [(exe, mode, 6 * e + j, name, F.LENGTHS[j % 3], j < 3) for e, name in enumerate(F.EFFECTS) for j in range(6)]
for a in jobs:
    if a[2] not in want: continue
    r = F.wet_case(a)
    print('#%-3d %-17s %2d s %-15s %s pieces %d/%d [%s] gain %5.2f delay %5.1f..%5.1f %-8s | out %6.1f %s%s | onto %6.1f %s | %.1f s' % (r['i'], r['effect'], r['seconds'], r['sound'], 'found  ' if r['found'] else 'nothing', r['shown'], r['sounding'],
          ''.join('x' if p['found'] else '.' for p in r['parts']), r['g'], r['dmin'], r['dmax'], ('%.2f Hz' % r['rate']) if r['periodic'] else 'free', r['out'], ' '.join('%.0f' % v for v in r.get('out_blocks', [])),
          ' (off half %.0f dB)' % r['out_off'] if 'out_off' in r else '', r['onto'], ' '.join('%.0f' % v for v in r.get('onto_blocks', [])), r['took']), flush=True)
    print('     live per part: %s | par %s' % (' '.join('%.2f' % v for v in r.get('live_parts', [])), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r['par'].items()}), flush=True)
    for p in r['parts']:
        if p['found'] and p.get('w'): print('     part %.1f-%.1f s seen %.2f g %.2f there %.2f | w per half second: %s' % (p['a'] / 48000, p['b'] / 48000, p.get('seen', 0), p['g'], p.get('on', 1.0), ' '.join('%.2f' % v if v >= 0 else '  - ' for v in p['w'])), flush=True); print('          there: ' + ' '.join('%s %.1f-%.1f' % ('LR'[c], a, b) for c, a, b in p.get('spans', [])), flush=True)
