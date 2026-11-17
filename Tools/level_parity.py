# The level-movement engine in C++ (Source/Core/Level.h, through drlevel) against the Python prototype it was ported from
# (Research/level/level.py): the same answer, the same measures, the same sounds?
#   python3 Tools/level_parity.py path/to/drlevel a.wav [b.wav ...]            every sound: find, take out, put onto the next sound in the list
#   python3 Tools/level_parity.py path/to/drlevel --list file.txt               the same, one path per line
# Prints one line per sound and a summary; exit code 1 if an answer ("is there a movement, which kind") differs.
import sys, os, json, subprocess, tempfile, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'Research', 'level'))
import level

def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def rel(a, b): return float(abs(a - b) / max(abs(a), abs(b), 1e-12))
def db(y, ref, x): return float(10 * np.log10(((y - ref) ** 2).mean() / max((x ** 2).mean(), 1e-30) + 1e-30))
def f64(p, n): return np.fromfile(p, dtype=np.float64).reshape(-1, 2)[:n]

TOG = ['P', 'r2', 'depth', 'skipfrac', 'up', 'steep', 'fund', 'rise', 'fall', 'climb', 'flat', 'share', 'excess', 'slope_med', 'slope_lo', 'slope_hi', 'slope_min', 'slope_max', 'corr_med', 'corr_lo', 'corr_min',
       'cyc_med', 'cyc_lo', 'cyc_hi', 'halves', 'r2c', 'silent', 'dis_med', 'dis_hi']
PAN = ['P', 'centre', 'swing', 'share', 'adj', 'resid', 'cycles', 'slope_med', 'slope_lo', 'slope_hi', 'known', 'fund', 'steep']

def main():
    exe = sys.argv[1]; paths = sys.argv[2:]
    if paths and paths[0] == '--list': paths = [l.strip() for l in open(paths[1]) if l.strip()]
    tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); bad = 0; worst = dict(measure=0.0, score=0.0, shape=0.0, sound=-999.0); nfound = 0
    for i, p in enumerate(paths):
        x = rd(p); other = rd(paths[(i + 1) % len(paths)]); info = {}; F = level.find(x, info=info)
        C = json.loads(subprocess.run([exe, 'measure', p], capture_output=True, text=True, check=True).stdout); line = '%-44s' % os.path.basename(p)[:44]
        # the answer
        kind = F['kind'] if F is not None else ''
        if (F is not None) != C['found'] or kind != C['kind']: bad += 1; line += ' ANSWER DIFFERS: Python %s, C++ %s' % (kind or 'nothing', C['kind'] or 'nothing')
        else: line += ' %-8s' % (kind or 'nothing')
        # the pan measures and the candidates' measures
        dm = 0.0; ds = 0.0; pan = info.get('pan'); cp = C['pan']
        if (pan is not None) != cp['ok']: line += ' | pan measured: Python %s C++ %s' % (pan is not None, cp['ok']); bad += 1
        elif pan is not None:
            for k in PAN: dm = max(dm, rel(pan[k], cp[k]))
            sp = level.score_pan(pan)
            if np.isfinite(sp) != (cp['score'] is not None): line += ' | pan score: Python %s C++ %s' % (sp, cp['score'])
            elif np.isfinite(sp): ds = max(ds, abs(sp - cp['score']))
        tg = info.get('together')
        if tg is not None:
            if len(tg) != len(C['cands']): line += ' | candidates: Python %d C++ %d' % (len(tg), len(C['cands'])); bad += 1
            else:
                for m, c in zip(tg, C['cands']):
                    for k in TOG: dm = max(dm, rel(m[k], c[k]))
                    if m['K'] != c['K'] or m['start'] != c['start'] or list(m['skip']) != c['skip'] or m['cyc_n'] != c['cyc_n'] or m['bands'] != c['bands']: line += ' | candidate at %.1f frames: counts differ' % m['P']; bad += 1
                    dm = max(dm, float(np.abs(np.asarray(m['g']) - np.asarray(c['g'])).max()) / max(float(np.abs(m['g']).max()), 1e-9))
                    s = level.score_together(m)
                    if np.isfinite(s) != (c['score'] is not None): line += ' | score at %.1f frames: Python %s C++ %s' % (m['P'], s, c['score'])
                    elif np.isfinite(s): ds = max(ds, abs(s - c['score']))
        line += ' | measures %.1e scores %.1e' % (dm, ds); worst['measure'] = max(worst['measure'], dm); worst['score'] = max(worst['score'], ds)
        # what was found, and the sounds
        if F is not None and C['found'] and kind == C['kind']:
            nfound += 1; prof = np.stack([C['profL'], C['profR']], 1); dsh = float(np.abs(prof - F['prof']).max()); dp = rel(F['period'], C['period'])
            ne = len(F.get('edges') or []); same = ne == len(C['edges']) and all(list(F.get(k) or []) == C[k] for k in ('skip', 'open')) and F['K'] == C['K']
            if not same: line += ' | edges / left-out steps differ (Python %d edges, C++ %d)' % (ne, len(C['edges'])); bad += 1
            else:
                for a, b in zip(F.get('edges') or [], C['edges']): dsh = max(dsh, abs(a['off'] - b['off']), float(np.abs(np.asarray(a['u']) - np.asarray(b['u'])).max()))
            if 'trust' in F: dsh = max(dsh, abs(F['trust'] - C['trust']))
            o = os.path.join(tmp, 'o.f64'); worstdb = -999.0
            for s in (-1.0, -0.5, 0.5):
                subprocess.run([exe, 'out', p, o, 's=%g' % s], capture_output=True, check=True); worstdb = max(worstdb, db(f64(o, len(x)), level.apply(x, F, s), x))
            for s in (1.0, 2.0):
                subprocess.run([exe, 'onto', p, paths[(i + 1) % len(paths)], o, 's=%g' % s], capture_output=True, check=True); worstdb = max(worstdb, db(f64(o, len(other)), level.apply(other, F, s), other))
            line += ' | period %.1e shape %.1e dB | sounds %.0f dB' % (dp, dsh, worstdb); worst['shape'] = max(worst['shape'], dsh); worst['sound'] = max(worst['sound'], worstdb)
        print(line + '   [C++ %.2f s]' % C['took'], flush=True)
    print('%d sounds, %d with a movement, %d differences in the answers or counts. Largest difference: measures %.1e (relative), scores %.1e, shapes %.1e dB, sounds %.0f dB against the sound'
          % (len(paths), nfound, bad, worst['measure'], worst['score'], worst['shape'], worst['sound']))
    sys.exit(1 if bad else 0)
main()
