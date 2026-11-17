# The echo engine in the app against its Python prototype (Research/echo/echo2.py), on the same sounds:
#   python3 Tools/echo_parity.py path/to/drecho sound.wav [sound.wav ...]
# Per sound: every lag that stood out and everything measured about the most echo-like ones, side by side; prints the
# largest differences. With --render it also compares the echoes taken out and put on (as a level under the sound, dB).
import sys, os, json, subprocess, tempfile, pickle, numpy as np, soundfile as sf
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, '..', 'Research', 'echo')); import echo2
args = [a for a in sys.argv[1:] if not a.startswith('--')]; tool, files = args[0], args[1:]; render = '--render' in sys.argv
rule = pickle.load(open(os.environ.get('RULE', os.path.join(here, '..', 'Research', 'echo', 'rule5.pkl')), 'rb')) if os.path.exists(os.environ.get('RULE', os.path.join(here, '..', 'Research', 'echo', 'rule5.pkl'))) else None
KEYS = ['ms', 'score', 'a', 'h', 'a2', 'added', 'removed', 'even', 'bands', 'others', 'npk', 'secs', 'pshare', 'pmed', 'piqr', 'pup', 'pn', 'pinfo', 'peff']
worst = {k: 0.0 for k in KEYS + ['p', 'cand.score', 'cand.a']}; bad = 0; nrec = 0
def rel(a, b): return abs(a - b) / max(abs(a), abs(b), 1e-9)
for f in files:
    x, sr = sf.read(f, dtype='float64', always_2d=True); x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1); assert sr == 48000
    C = json.loads(subprocess.run([tool, 'measure', f], capture_output=True, text=True, check=True).stdout)
    s, n = echo2.stretch(x); msg = []
    if (s, n) != (C['start'], C['length']): msg.append('stretch %s against %s' % ((s, n), (C['start'], C['length'])))
    E = []
    for ch in range(2):
        cand, recs = echo2.measure(x[s:s + n, ch]); cc = C['ch'][ch]; E.append(recs)
        if [k['q'] for k in cand] != [k['q'] for k in cc['cand']] or [bool(k['rah']) for k in cand] != [k['rah'] for k in cc['cand']]: msg.append('ch %d: lags that stand out differ: %s against %s' % (ch, [(k['q'], bool(k['rah'])) for k in cand], [(k['q'], k['rah']) for k in cc['cand']]))
        else:
            for a, b in zip(cand, cc['cand']): worst['cand.score'] = max(worst['cand.score'], rel(a['score'], b['score'])); worst['cand.a'] = max(worst['cand.a'], rel(a['a'], b['a']))
        if [r['q'] for r, _ in recs] != [r['q'] for r in cc['recs']]: msg.append('ch %d: measured lags differ: %s against %s' % (ch, [r['q'] for r, _ in recs], [r['q'] for r in cc['recs']])); continue
        for (a, _), b in zip(recs, cc['recs']):
            nrec += 1
            for k in KEYS:
                d = rel(a[k], b[k]) if k not in ('bands', 'others', 'npk', 'pn') else float(a[k] != b[k]); worst[k] = max(worst[k], d)
                if d > 1e-6: msg.append('ch %d lag %d: %s %.9g against %.9g' % (ch, a['q'], k, a[k], b[k]))
            if rule is not None:
                p = float(rule.predict_proba(np.array([echo2.feats(a)], float))[0, 1]); worst['p'] = max(worst['p'], abs(p - b['p']))
                if abs(p - b['p']) > 1e-6: msg.append('ch %d lag %d: p %.6f against %.6f' % (ch, a['q'], p, b['p']))
    line = '%-40s %d + %d lags measured, found %s, sure %.3f, %.2f s in the app' % (os.path.basename(f)[:40], len(C['ch'][0]['recs']), len(C['ch'][1]['recs']), [c['found']['q'] if c['found'] else None for c in C['ch']], C['sure'], C['took'])
    if render and any(c['found'] for c in C['ch']):
        # the found echoes as the prototype takes them out and puts them on (the same lags as the app found)
        Ep = []
        for ch in range(2):
            fq = C['ch'][ch]['found']; e = next((e for r, e in E[ch] if fq and r['q'] == fq['q']), None); Ep.append(e)
        with tempfile.TemporaryDirectory() as td:
            for mode, s_ in (('out', -1.0), ('onto', 1.0)):
                o = os.path.join(td, mode + '.wav'); subprocess.run([tool, mode, f] + ([f] if mode == 'onto' else []) + [o], capture_output=True, check=True)
                y, _ = sf.read(o, dtype='float64', always_2d=True); ref = echo2.apply(x, Ep, s_)[:len(x)]
                d = 10 * np.log10(max(((y - ref) ** 2).sum(), 1e-30) / max((ref ** 2).sum(), 1e-30)); line += ' | %s: %.1f dB' % (mode, d)
    print(line + ('' if not msg else '\n    ' + '\n    '.join(msg[:8]))); bad += bool(msg)
print('\n%d sounds, %d with a difference; %d measured lags. Largest relative differences: %s' % (len(files), bad, nrec, ', '.join('%s %.1e' % (k, v) for k, v in worst.items())))
