# The candidate search (ring.spectrum_peaks): is the tone's own line (at f or 2 f) among the candidates, on every ring case at hand,
# and on every clip of the corpus with a tone put on? How many candidates do the clean clips give?   python3 rk1.py
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import ring as E, rf1
def candidates(pw): return E.spectrum_peaks(pw)
def has(c, f): return any(abs(v - f) < 0.5 or abs(v - 2 * f) < 0.5 for v in c)
if __name__ == '__main__':
    bad = 0; tot = 0; t0 = time.time()
    meta = json.load(open('genr/meta.json')); rows = {json.loads(l)['name']: json.loads(l) for fn in sorted(glob.glob('rt7_[0-9].jsonl')) for l in open(fn) if l.strip()}
    for nm in sorted(meta):
        wet = truth.rd('genr/' + nm + '.flac') / meta[nm]['gain']; c = candidates(E.power(wet)); f = rows[nm]['true_f']; tot += 1
        if not has(c, f): bad += 1; print('MISSING', nm, 'true f %.2f rho %.2f' % (f, rows[nm]['true_rho']), ['%.1f' % v for v in c])
    print('fresh cases: %d, the tone not among the candidates: %d' % (tot, bad)); bad = tot = 0
    for clip, f, rho, ph in rf1.cases(40):
        x = truth.rd('/tmp/claude-0/v2/corpus/dry/' + clip + '.wav'); c = candidates(E.power(rf1.plain(x, f, rho, ph))); tot += 1
        if not has(c, f): bad += 1; print('MISSING', clip, f, rho, ['%.1f' % v for v in c])
    print('round pitches: %d, missing: %d' % (tot, bad)); bad = tot = 0
    cm = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); rng = np.random.default_rng(5); nc = []
    for clip in sorted(cm):
        x = truth.rd('/tmp/claude-0/v2/corpus/dry/' + clip + '.wav'); nc.append(len(candidates(E.power(x))))
        for rho in (0.0, float(rng.uniform(0.2, 1.4))):
            f = float(np.exp(rng.uniform(np.log(25.0), np.log(4500.0)))); c = candidates(E.power(rf1.plain(x, f, rho, float(rng.uniform(0, 6.28))))); tot += 1
            if not has(c, f): bad += 1; print('MISSING', clip if not clip.startswith('real_') else 'real_XX', 'f %.1f rho %.2f' % (f, rho), ['%.1f' % v for v in c])
    print('every clip of the corpus with a tone put on (twice): %d, missing: %d; candidates on the clean clips: none on %d of %d, mean %.1f; %.0f s' % (tot, bad, sum(v == 0 for v in nc), len(nc), np.mean(nc), time.time() - t0))
