# Clean sounds must come out clean: find() on every dry corpus sound (198, split dev / test); what is reported where nothing was put on.
#   python3 clean.py   -> clean_<TAG>.jsonl      (of real recordings only the corpus id and the measured numbers are kept)
import sys, os, json, time, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/level'); import level
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); TAG = os.environ.get('TAG', 'v1'); out = open('/tmp/claude-0/v2/level/clean_%s.jsonl' % TAG, 'w'); n = {'dev': 0, 'test': 0}; f = {'dev': 0, 'test': 0}
for cid in sorted(meta):
    x, sr = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % cid, dtype='float64', always_2d=True); x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1); sp = meta[cid]['split']; info = {}
    F = level.find(x, info=info); n[sp] += 1; row = dict(id=cid, kind=meta[cid]['kind'], split=sp, found=F is not None)
    tg = [m for m in info.get('together', []) if np.isfinite(level.score_together(m))]; row['best_together'] = max([level.score_together(m) for m in tg], default=None); row['pan_score'] = (level.score_pan(info['pan']) if info.get('pan') else None)
    if row['pan_score'] is not None and not np.isfinite(row['pan_score']): row['pan_score'] = None
    if F is not None:
        f[sp] += 1; y = level.apply(x, F, -1.0); ch = 20 * np.log10(np.sqrt(((y - x) ** 2).mean()) / max(np.sqrt((x ** 2).mean()), 1e-12) + 1e-12)
        row.update(what=F['kind'], rate=F['rate'], depth=F['depth'], change_db=ch, measures={k: v for k, v in F['measures'].items() if not isinstance(v, (list, np.ndarray))})
        print('%-28s %-16s %-4s reported: %s %.2f Hz, %.1f dB deep; taking it out changes the sound by %.1f dB of its level' % (cid, meta[cid]['kind'], sp, F['kind'], F['rate'], F['depth'], ch), flush=True)
    out.write(json.dumps(row, default=float) + '\n'); out.flush()
print('reported on clean sounds: dev %d of %d, test %d of %d' % (f['dev'], n['dev'], f['test'], n['test']))
