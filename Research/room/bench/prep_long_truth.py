# The true direct part of the long sounds with reverb (prep_long.py: loopr, catr), for scoring how well the app takes the
# reverb out of a long sound. As the scoreboard does it (mkbench.py, proto/truth.py): the tool's responses to a click on
# the left and on the right channel give the direct arrival (1 ms before to 2.5 ms after the strongest one), and the dry
# sound through that part alone is the direct part of the wet sound.
#   python3 prep_long_truth.py [processes]     -> wav/long_truth/<name>.flac (24 bit, on the scale of the wet file), long_truth.json
import sys, os, json, hashlib
sys.dont_write_bytecode = True; sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/roomsure')
import numpy as np, soundfile as sf, scipy.signal as ss
import zoo as R, prep_long as PL
from multiprocessing import Pool
SR = 48000; OUT = '/tmp/claude-0/v2/roomsure/wav/long_truth/'; os.makedirs(OUT, exist_ok=True); CLICK = 480
def one(r):
    b = os.path.basename(r['path'])[:-4]; kind = r['kind']; name = r['tool']; i = int(b[-2:])
    try:
        rng = PL.rng_of(kind, name, i)
        if kind == 'loopr': a = str(rng.choice(PL.NAMES[r['split']])); assert a == r['dry']; x = np.tile(PL.dry(a), (PL.TIMES, 1))
        else: x = sf.read(PL.MVL + r['dry'] + '.wav', always_2d=True)[0].astype(np.float64)
        p = R.ZOO[name][2](rng); w = R.apply(name, x, p, 3.0); pk = np.abs(w).max(); g = 0.98 / pk if pk >= 0.98 else 1.0
        saved = sf.read(r['path'], always_2d=True)[0].astype(np.float64); assert len(saved) == len(w)
        err = 10 * np.log10(((saved - g * w) ** 2).sum() / (saved ** 2).sum() + 1e-30)          # the file against the sound made again: only the 16 bit rounding may differ
        cl = R.impulse(SR * 6, CLICK); cl[:, 1] = 0; hl = R.apply(name, cl, p)[CLICK:]; cr = R.impulse(SR * 6, CLICK); cr[:, 0] = 0; hr = R.apply(name, cr, p)[CLICK:]
        e = (hl ** 2).sum(1) + (hr ** 2).sum(1); k = int(np.argmax(e)); a0, b0 = max(0, k - 48), k + 120; n = len(w)
        direct = np.stack([ss.fftconvolve(x[:, 0], hl[a0:b0, o])[:n] + ss.fftconvolve(x[:, 1], hr[a0:b0, o])[:n] for o in range(2)], 1)
        if a0 > 0: direct = np.vstack([np.zeros((a0, 2)), direct])[:n]
        if len(direct) < n: direct = np.vstack([direct, np.zeros((n - len(direct), 2))])
        # how well the click responses describe what the tool did (exact for a fixed linear tool)
        pred = np.stack([ss.fftconvolve(x[:, 0], hl[:, o])[:n] + ss.fftconvolve(x[:, 1], hr[:, o])[:n] for o in range(2)], 1); pred = np.vstack([pred, np.zeros((max(0, n - len(pred)), 2))])[:n]
        fixed = 10 * np.log10(((w - pred) ** 2).sum() / (w ** 2).sum() + 1e-30)
        rev = w - direct; drr = 10 * np.log10(max((direct ** 2).sum(), 1e-20) / max((rev ** 2).sum(), 1e-20))
        sf.write(OUT + b + '.flac', direct * g, SR, format='FLAC', subtype='PCM_24')
        return dict(name=b, path=r['path'], truth=OUT + b + '.flac', kind=kind, tool=name, split=r['split'], gain=float(g), file_err_db=float(err), fixed_db=float(fixed), drr=float(drr), dry_sec=len(x) / SR, sec=n / SR)
    except Exception as ex:
        print('FAILED', b, str(ex)[:200], flush=True); return None
if __name__ == '__main__':
    rows = [r for r in json.load(open('/tmp/claude-0/v2/roomsure/long_rows.json')) if r['wet'] == 1 and not int(r['cut'])]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as p: out = [o for o in p.map(one, rows, chunksize=1) if o is not None]
    json.dump(out, open('/tmp/claude-0/v2/roomsure/long_truth.json', 'w'), indent=1)
    print(len(rows), 'sounds,', len(out), 'truths; file against the sound made again: worst %.0f dB; click responses describe the tool: median %.0f dB, worst %.0f dB; direct to reverb: %.1f .. %.1f dB' % (
        max(o['file_err_db'] for o in out), np.median([o['fixed_db'] for o in out]), max(o['fixed_db'] for o in out), min(o['drr'] for o in out), max(o['drr'] for o in out)))
