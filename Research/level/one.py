# single cases of long.py's sets with a given drlevel: python3 one.py path/to/drlevel wet|part|drift|slow i [i ...]
import sys, json, numpy as np
import long as L
exe, mode = sys.argv[1], sys.argv[2]; want = [int(v) for v in sys.argv[3:]]
if mode == 'wet': jobs = [(exe, mode, 8 * e + j, name, L.LENGTHS[j % 4], j < 4) for e, name in enumerate(L.EFFECTS) for j in range(8)]
elif mode == 'part': jobs = [(exe, mode, 4 * e + j, name, L.LENGTHS[1 + j % 3], j < 2) for e, name in enumerate(L.EFFECTS) for j in range(4)]
elif mode == 'drift': jobs = [(exe, mode, 8 * e + j, name, L.LENGTHS[j % 4], j < 4) for e, name in enumerate(('trem', 'pan')) for j in range(8)]
elif mode == 'slow': jobs = [(exe, mode, 8 * e + j, name, L.LENGTHS[j % 4], j < 4) for e, name in enumerate(('pan_sox', 'pan_tap', 'pan_calf')) for j in range(8)]
for a in jobs:
    if a[2] not in want: continue
    r = L.wet_case(a)
    print('#%-3d %-16s %2d s %-8s score %s rate %.3f Hz (%s) depth %5.1f pieces %d/%d%s | out %6.1f %s%s | onto %6.1f | spans %s | %.1f s' % (r['i'], r['effect'], r['seconds'], r['kind'] if r['found'] else 'nothing', '%5.2f' % r['score'] if r['found'] else '-', r['rate'],
          ' '.join('%s=%.3g' % (k, v) for k, v in r['par'].items() if k in ('speed', 'hz', 'freq', 'rate', 'auto')), r['depth'], r['carrying'], r['sounding'], ' long look' if r.get('wide') else '', r['out'], ' '.join('%.0f' % v for v in r.get('out_blocks', [])),
          ' (off half %.0f dB)' % r['out_off'] if 'out_off' in r else '', r['onto'], ' '.join('%.1f-%.1f' % (a / 48000, b / 48000) for a, b in r.get('spans', [])), r['took']), flush=True)
