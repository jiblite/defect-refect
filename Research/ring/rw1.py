# The level window: eight cycles of the tone (a power of two, 21 ms at least) against whole cycles of it (5 ms at least),
# and which quiet samples count. Per case: found, the match (limit 1 dB), what is left.   python3 rw1.py
import sys, os, json, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import ring as E, rf1
old_window = E.window
def whole(f, least=float(os.environ.get('LEAST', 240))):
    P = SR / f; k = int(np.ceil(least / P)); return 2 * int(round(k * P))
def run(y, x, tag):
    out = []
    for name, win, gate in (('8 cycles, -80 dB', old_window, 1e-8), ('8 cycles, -50 dB', old_window, 1e-5), ('whole cycles, -80 dB', whole, 1e-8), ('whole cycles, -50 dB', whole, 1e-5)):
        E.window = win; E.GATE = gate; t0 = time.time(); r = E.find(y); dt = time.time() - t0
        if 'f' in r:
            left = 0.0
            if r['found'] and x is not None: k = float((r['x'] * x).sum() / (x ** 2).sum()); left = db(r['x'] - k * x, k * x)
            out.append('%-22s found %d f %.5f rho %.4f floor %.1e win %.3f match %.2f dip %.1f | left %6.1f dB | %d passes %.1f s' % (name, r['found'], r['f'], r['rho'], r['eps'], r['win'], r['match'], r['dip'], left, r['passes'], dt))
        else: out.append('%-22s not measured (first win %.3f)' % (name, r.get('first_win', -9)))
    E.window = old_window; E.GATE = 1e-5
    print(tag); [print('    ' + l) for l in out]; sys.stdout.flush()
meta = json.load(open('genr/meta.json')); rows = {json.loads(l)['name']: json.loads(l) for fn in sorted(glob.glob('rt7_[0-9].jsonl')) for l in open(fn) if l.strip()}
for nm in (sys.argv[1:] or ['rng_sox_07', 'rng_calf_04', 'plain_13', 'plain_05', 'rng_calf_01', 'rng_mda_00', 'rng_swh_03']):
    m = meta[nm]; wet = truth.rd('genr/' + nm + '.flac') / m['gain']; dry = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['clip'] + '.wav'); n = min(len(wet), len(dry)); run(wet[:n], dry[:n], '%s true f %.3f rho %.3f' % (nm, rows[nm]['true_f'], rows[nm]['true_rho']))
if not sys.argv[1:]:
    for clip, f, rho, ph in (('sp_loop_safari', 35.0, 0.0, 1.0), ('fs_stab_woodblock', 300.0, 0.0, 0.0016), ('sp_loop_breakbeat', 28.0, 0.4, 2.0), ('lm_seq_drums_3', 45.0, 0.0, 0.5), ('fs_stab_woodblock', 60.0, 0.0, 0.3)):
        x = truth.rd('/tmp/claude-0/v2/corpus/dry/' + clip + '.wav'); run(rf1.plain(x, f, rho, ph), x, '%s plain f %.1f rho %.1f' % (clip, f, rho))
