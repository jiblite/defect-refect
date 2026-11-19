# The sides engine in C++ (Source/Core/Sides.h, through drsides) against the Python prototype it was ported from
# (Research/sides/sides.py): the same verdict, the same taps, the same sounds?
#   python3 Tools/sides_parity.py path/to/drsides a.wav b.wav [a2.wav b2.wav ...]
# Every pair: what was done between the two sides is read off a, taken out of it, and put onto b. Prints the numbers
# side by side; exit code 1 if a verdict (kind, read as the pair or as the copy, which taps) differs or a sound is
# further than -100 dB from the prototype's.
import sys, os, json, subprocess, tempfile, time, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'sides')]
import sides as E

def db(a, b): return 10 * np.log10(((a - b) ** 2).mean() / max((b ** 2).mean(), 1e-30) + 1e-30)
def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def taps(v): return [(int(i), float(v[i])) for i in np.where(v != 0)[0]]

def main():
    exe = sys.argv[1]; paths = sys.argv[2:]; tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); pre = os.path.join(tmp, 'x'); bad = 0; worst = -999.0
    for pa, pb in zip(paths[0::2], paths[1::2]):
        a = rd(pa); b = rd(pb); t0 = time.time(); r = subprocess.run([exe, pa, pb, pre], capture_output=True, text=True); t1 = time.time()
        if r.returncode: print(r.stderr); bad += 1; continue
        J = json.loads(r.stdout.strip().splitlines()[-1]); t2 = time.time(); F = E.find(a); t3 = time.time()
        print('%s: app %.2f s (find %.2f s), prototype %.1f s | %s' % (os.path.basename(pa), t1 - t0, J['took'], t3 - t2, ' '.join(r.stderr.split())), flush=True)
        same = F['kind'] == J['kind'] and F['mode'] == J['mode'] and ('pair' in F) == J['has_pair'] and ('copy' in F) == J['has_copy']
        print('  prototype: %s %s | app: %s %s' % (F['kind'], F['mode'], J['kind'], J['mode']))
        if 'pair' in F and J['has_pair']:
            p, sh, ck = F['pair'], F['shape'], F['pair_check']; ta, tb = taps(p['a']), taps(p['b']); la = [i for i, v in ta] == [i for i, v in J['a']] and [i for i, v in tb] == [i for i, v in J['b']]
            dv = max([abs(v - w[1]) for (i, v), w in zip(ta + tb, J['a'] + J['b'])] or [0.0]) if la else float('nan'); same = same and la and dv < 1e-6 and p['way'] == J['way']
            print('  the pair: way %s / %s, taps %d / %d at the same places: %s, largest difference of a tap %.1e' % (p['way'], J['way'], sh['taps'], J['taps'], 'yes' if la else 'NO', dv))
            print('     apart %d / %d  reach %d / %d  made %.3f / %.3f  covered %.5f / %.5f  inside %.3e / %.3e  beyond %.3f / %.3f  res %.3e / %.3e  fits %.5f / %.5f' % (sh['apart'], J['apart'], sh['reach'], J['reach'], sh['made'], J['made'], ck['covered'], J['covered'], ck['inside'], J['inside'], ck['beyond'], J['beyond'], p['res'], J['res'], p['fits'], J['fits']))
        if 'copy' in F and J['has_copy']:
            g = F['copy']; print('  the copy: share %.5f / %.5f  all %.5f / %.5f  balance %.5f / %.5f  side %.3f / %.3f dB' % (g['share'], J['share'], g['all'], J['all'], g['balance'], J['balance'], g['side_db'], J['side_db'])); same = same and abs(g['share'] - J['share']) < 1e-6
            if 'copy_check' in F and J['copy_checked']: print('     covered %.5f / %.5f  inside %.3e / %.3e' % (F['copy_check']['covered'], J['copy_covered'], F['copy_check']['inside'], J['copy_inside']))
        if F['mode'] is not None and J['mode'] is not None: same = same and [list(s) for s in F['spans']] == J['spans']; print('  spans: %d / %d' % (len(F['spans']), len(J['spans'])))
        bad += int(not same)
        if not same: print('  DIFFERENT ANSWERS')
        if not (F['mode'] is not None and J['mode'] is not None and same): continue
        x = F['pair_check']['x'] if F['mode'] == 'pair' else F['copy_check']['copy']; d = db(np.fromfile(pre + '.x.f64'), x); worst = max(worst, d); print('  %-6s app against prototype: %.1f dB' % ('x', d))
        d1 = float(np.abs(np.fromfile(pre + '.same.f64').reshape(2, len(a)).T - a).max()); print('  Amount in the middle: the largest difference from the sound itself %.1e' % d1); bad += int(d1 != 0.0)
        for tag, ref in (('out', E.own(a, F, 0.0)), ('half', E.own(a, F, 0.5)), ('twice', E.own(a, F, 2.0)), ('onto', E.onto(b, F, 1.0)), ('onto2', E.onto(b, F, 2.0)), ('onto05', E.onto(b, F, 0.5))):
            d = db(np.fromfile(pre + '.%s.f64' % tag).reshape(2, len(ref)).T, ref); worst = max(worst, d); print('  %-6s app against prototype: %.1f dB' % (tag, d), flush=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp); print('%d pairs, %d with different answers; the sounds differ by at most %.0f dB' % (len(paths) // 2, bad, worst)); sys.exit(1 if bad or worst > -100.0 else 0)
main()
