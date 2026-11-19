# Every clip of the corpus with a plain ring modulation put on, twice (once with no plain sound beside the tone, once
# with some), through the app's own code (drring, Tests/ring.cpp): is the tone found, how exactly, and what is left of
# it once it is taken out? The same draws as rk1.py (which only asks whether the tone is among the candidates).
#   python3 rk2.py run i n    -> rk2_<i>.jsonl
#   python3 rk2.py show
import sys, os, json, glob, subprocess, tempfile, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import rf1
BIN = os.environ.get('BIN', '/tmp/claude-0/drring'); HERE = '/tmp/claude-0/v2/newfx/'
def cases():
    cm = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); rng = np.random.default_rng(5); out = []
    for clip in sorted(cm):
        for rho in (0.0, float(rng.uniform(0.2, 1.4))):
            f = float(np.exp(rng.uniform(np.log(25.0), np.log(4500.0)))); out.append((clip, f, rho, float(rng.uniform(0, 6.28))))
    return out
if len(sys.argv) > 1 and sys.argv[1] == "run" and __name__ == "__main__":
    i0, n0 = int(sys.argv[2]), int(sys.argv[3]); out = open(HERE + 'rk2_%d.jsonl' % i0, 'w'); tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); pa = os.path.join(tmp, 'a.wav'); pre = os.path.join(tmp, 'o')
    for k, (clip, f, rho, ph) in list(enumerate(cases()))[i0::n0]:
        x = truth.rd('/tmp/claude-0/v2/corpus/dry/' + clip + '.wav'); y = rf1.plain(x, f, rho, ph); sf.write(pa, y, SR, subtype='FLOAT')
        r = subprocess.run([BIN, pa, pa, pre], capture_output=True, text=True); row = dict(k=k, clip=clip, true_f=f, true_rho=rho, true_phase=ph, seconds=len(x) / SR)
        if r.returncode: row['error'] = r.stderr[-200:]
        else:
            J = json.loads(r.stdout.strip().splitlines()[-1]); row.update(J)
            if J['found']:
                o = np.fromfile(pre + '.out.f64').reshape(2, -1).T; kk = float((o * x).sum() / max(float((x ** 2).sum()), 1e-30)); row['left'] = db(o - kk * x, kk * x); row['out'] = score.removed(x, y, o)[0]
        out.write(json.dumps(row) + '\n'); out.flush()
    for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
    os.rmdir(tmp)
elif __name__ == "__main__":
    rows = [json.loads(l) for fn in sorted(glob.glob(HERE + 'rk2_[0-9].jsonl')) for l in open(fn) if l.strip()]; fd = [r for r in rows if r.get('found')]
    def nm(r): return r['clip'] if not r['clip'].startswith('real_') else 'one of the recordings'
    print('%d clips with a tone put on: found %d, errors %d' % (len(rows), len(fd), sum('error' in r for r in rows)))
    for lo, hi in ((25, 50), (50, 100), (100, 400), (400, 1500), (1500, 4500)):
        q = [r for r in rows if lo <= r['true_f'] < hi]; g = [r for r in q if r.get('found')]
        print('   tone %4d to %4d Hz: %3d cases, found %3d' % (lo, hi, len(q), len(g)) + ('; taken out: median %.1f, least %.1f; left beside the dry sound: median %.1f dB, worst %.1f dB' % (np.median([r['out'] for r in g]), min(r['out'] for r in g), np.median([r['left'] for r in g]), max(r['left'] for r in g)) if g else ''))
    for a, b in ((0.0, 1e-9), (1e-9, 1.0), (1.0, 9.0)):
        q = [r for r in rows if a <= r['true_rho'] < b or (a == 0.0 and r['true_rho'] == 0.0)]; g = [r for r in q if r.get('found')]
        if q: print('   plain sound beside the tone %.1f to %.1f: %3d cases, found %3d' % (a, b, len(q), len(g)) + ('; taken out: median %.1f, least %.1f; left: median %.1f dB, worst %.1f dB' % (np.median([r['out'] for r in g]), min(r['out'] for r in g), np.median([r['left'] for r in g]), max(r['left'] for r in g)) if g else ''))
    if fd: print('   pitch off by at most %.5f Hz (median %.6f); seconds a sound: median %.2f, most %.2f' % (max(abs(r['f'] - r['true_f']) for r in fd), np.median([abs(r['f'] - r['true_f']) for r in fd]), np.median([r['took'] for r in rows if 'took' in r]), max(r['took'] for r in rows if 'took' in r)))
    why = {}
    for r in rows:
        if r.get('found') or 'error' in r: continue
        if not r.get('looked'): w = 'no candidate line'
        elif not r.get('measured'): w = 'first look says no'
        elif abs(r['f'] - r['true_f']) > 0.5: w = 'a wrong line measured'
        else: w = ' '.join(t for t, bad in (('match', r['match'] > 1.0), ('dip', r['dip'] < 10.0), ('offset', r['offset'] > -20.0), ('win', r['win'] < 0.3), ('holes', r['holes'] > 0.1)) if bad)
        why.setdefault(w, []).append(r)
    for w in sorted(why, key=lambda w: -len(why[w])):
        print('   NOT FOUND, %s: %d' % (w, len(why[w])))
        for r in why[w][:int(os.environ.get('SHOW', 12))]: print('       %-26s %5.1f s true f %8.2f rho %.2f | first win %.3f at %.2f Hz | f %.3f rho %.3f win %.2f match %.2f dip %.1f offset %.1f holes %.3f' % (nm(r), r['seconds'], r['true_f'], r['true_rho'], r.get('first_win', 0), r.get('first_f', 0), r.get('f', 0), r.get('rho', 0), r.get('win', 0), r.get('match', 0), r.get('dip', 0), r.get('offset', 0), r.get('holes', 0)))
    low = sorted([r for r in fd if r['out'] < 90 or abs(r['f'] - r['true_f']) > 0.01], key=lambda r: r['out'])
    print('   found but taken out under 90 or pitch off by over 0.01 Hz: %d' % len(low))
    for r in low[:int(os.environ.get('SHOW', 12))]: print('       %-26s %5.1f s true f %8.2f rho %.2f | f %.4f rho %.3f floor %.1e match %.2f dip %.1f | out %.1f, left %.1f dB' % (nm(r), r['seconds'], r['true_f'], r['true_rho'], r['f'], r['rho'], r['eps'], r['match'], r['dip'], r['out'], r['left']))
