# The level-movement engine on long sounds, as the app measures them (analyse / ownDb / ontoDb in Source/Core/Level.h,
# through Tests/level.cpp = drlevel). Long sounds are made by joining clean 6 s corpus clips (30 ms blends, the first
# clip entered at a random place, so that the joints never fall on the engine's piece borders).
#   python3 long.py path/to/drlevel clean [out.jsonl]     long clean sounds: is a movement reported?
#   python3 long.py path/to/drlevel wet [out.jsonl]       long sounds through outside tremolos, auto pans and gates: found? taken out? put on a second long sound?
#   python3 long.py path/to/drlevel part [out.jsonl]      the effect on one half of the sound only
#   python3 long.py path/to/drlevel drift [out.jsonl]     a tremolo or pan whose speed wanders by 3 % either way
#   python3 long.py path/to/drlevel slow [out.jsonl]      pans of 0.03 to 0.3 Hz
#   python3 long.py --show file.jsonl [...]
# SEED=1 python3 long.py ...: a second draw of sounds and settings (kept for the final check). CARRY, PRICE, LEAN: the
# engine's thresholds, for trying others. Real recordings are
# named by their corpus ids only.
import sys, os, json, subprocess, tempfile, multiprocessing as mp
import numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/detect/py'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
SR = 48000; CORPUS = '/tmp/claude-0/v2/corpus'; LENGTHS = (20, 33, 47, 62); SEED = int(os.environ.get('SEED', '0')) * 100000
EFFECTS = ('trm_sox', 'trm_calf', 'trm_tap', 'trm_gx', 'pan_tap', 'pan_calf', 'pan_sox', 'pan_mda', 'gat_sox', 'gat_calf', 'gat_ffmpeg_keyed')
META = json.load(open(os.path.join(CORPUS, 'meta.json'))); CARRY = os.environ.get('CARRY'); PRICE = os.environ.get('PRICE'); LEAN = os.environ.get('LEAN')

def clip(name):
    x, sr = sf.read(os.path.join(CORPUS, 'dry', name + ('' if name.endswith('.wav') else '.wav')), dtype='float64', always_2d=True); assert sr == SR
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def long_sound(rng, seconds, same_kind):
    """A long clean sound: clips of one kind (same_kind) or of any kind, joined."""
    names = sorted(META)
    if same_kind:
        kinds = sorted({META[n]['kind'] for n in names}); need = int(np.ceil(seconds / 6.0)) + 1
        kinds = [k for k in kinds if sum(META[n]['kind'] == k for n in names) >= need]; kind = kinds[int(rng.integers(len(kinds)))]; names = [n for n in names if META[n]['kind'] == kind]
    else: kind = 'mixed'
    order = [names[i] for i in rng.permutation(len(names))]; n = int(seconds * SR); X = int(0.03 * SR); out = np.zeros((0, 2)); used = []
    for j, nm in enumerate(order):
        c = clip(nm)
        if j == 0: c = c[int(rng.uniform(0.5, 3.0) * SR):]
        used.append(nm.replace('.wav', ''))
        if len(out) == 0: out = c
        else:
            w = np.sin(0.5 * np.pi * np.arange(X) / X)[:, None]; out = np.vstack([out[:-X], out[-X:] * np.sqrt(1 - w ** 2) + c[:X] * w, c[X:]])
        if len(out) >= n: break
    return out[:n], kind, used
def run(exe, args):
    r = subprocess.run([exe] + args + (['carry=' + CARRY] if CARRY else []) + (['price=' + PRICE] if PRICE else []) + (['lean=' + LEAN] if LEAN else []), capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-300:] + r.stderr[-300:])
    return json.loads(r.stdout)
def brief(M): return dict(found=M['found'], kind=M['kind'] if M['found'] else '', steps=M['steps'], wide=M.get('wide', False), score=M['score'] if M['found'] else None, sure=M['sure'], rate=M['rate'], depth=M['depth'], sounding=M['sounding'], carrying=M['carrying'],
                         lead=M['lead'], steady=M.get('steady'), knots=M.get('knots'), pieces=[dict(a=p['a'], b=p['b'], sounds=p['sounds'], found=p['found'], use=p['use'], guided=p.get('guided', False), clocked=p.get('clocked', False), there=p.get('there'), against=p['against'], own=p.get('own'), score=p.get('score'), rate=p.get('rate'), depth=p.get('depth')) for p in M['pieces']], windows=M.get('windows', []), spans=M.get('spans', []), took=M['took'])

def clean_case(a):
    exe, i, seconds, same = a; rng = np.random.default_rng(SEED + 7000 + i); x, kind, used = long_sound(rng, seconds, same)
    d = tempfile.mkdtemp(dir='/dev/shm'); p = os.path.join(d, 'a.wav'); sf.write(p, x, SR, subtype='FLOAT'); M = run(exe, ['pieces', p]); os.remove(p); os.rmdir(d)
    return dict(i=i, mode='clean', seconds=seconds, sound=kind, clips=used, **brief(M))

def blocks(ref, start, out, n, live=None):
    """The score in every 6 s stretch (the last one if it is at least 3 s long); live: only stretches where the effect is on."""
    import score
    v = []
    for a in range(0, n, 6 * SR):
        b = min(n, a + 6 * SR)
        if b - a >= 3 * SR and (live is None or live[a:b].mean() > 0.5): v.append(score.removed(ref[a:b], start[a:b], out[a:b])[0])
    return v
def wander(rng, n, f0):
    """A count of cycles whose speed wanders by 3 % either way, slowly (once in 23 to 41 s)."""
    t = np.arange(n) / SR; T = rng.uniform(23, 41); th = rng.uniform(0, 2 * np.pi)
    return f0 * (t - 0.03 * T / (2 * np.pi) * (np.cos(2 * np.pi * t / T + th) - np.cos(th))) + rng.uniform(0, 1)
def effect_of(mode, name, rng, n):
    """What is put on a dry sound: fn(dry) -> wet, the family, the settings, and where it is on (None: everywhere)."""
    import zoo_fx
    if mode == 'drift':
        f0 = float(np.exp(rng.uniform(np.log(0.4 if name == 'pan' else 1.5), np.log(3.0 if name == 'pan' else 12.0)))); d = float(rng.uniform(0.4, 0.9)); ph = wander(rng, n, f0)
        if name == 'pan': g = np.stack([1 + d * np.sin(2 * np.pi * ph), 1 - d * np.sin(2 * np.pi * ph)], 1) / (1 + d)
        else: g = np.repeat((1 - d * (0.5 - 0.5 * np.cos(2 * np.pi * ph)))[:, None], 2, 1)
        return (lambda x: x * g[:len(x)]), ('autopan' if name == 'pan' else 'tremolo'), dict(rate=f0, depth=d), None
    fam = zoo_fx.ZOO[name][0]; par = zoo_fx.ZOO[name][2](rng)
    if mode == 'slow':
        key = dict(pan_sox='rate', pan_tap='freq', pan_calf='hz')[name]; par[key] = float(np.exp(rng.uniform(np.log(0.03), np.log(0.3))))
    live = None
    if mode == 'part':
        live = np.zeros(n); cut = int(n * rng.uniform(0.4, 0.6)); X = int(0.05 * SR)
        if rng.random() < 0.5: live[:cut] = 1; live[cut:cut + X] = np.linspace(1, 0, X)
        else: live[cut:] = 1; live[cut - X:cut] = np.linspace(0, 1, X)
        def fn(x): w = zoo_fx.apply(name, x, par); m = live[:len(x), None]; return x * (1 - m) + w * m
    else: fn = lambda x: zoo_fx.apply(name, x, par)
    return fn, fam, {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in par.items()}, live
def wet_case(a):
    import score
    exe, mode, i, name, seconds, same = a; rng = np.random.default_rng(SEED + dict(wet=9000, part=11000, drift=13000, slow=15000)[mode] + i); n = int(seconds * SR)
    fn, fam, par, live = effect_of(mode, name, rng, n); dry, kind, used = long_sound(rng, seconds, same); dryb, kindb, usedb = long_sound(rng, seconds, same); wet = fn(dry); wetb = fn(dryb)
    d = tempfile.mkdtemp(dir='/dev/shm'); pa, pb, po, pq = (os.path.join(d, f) for f in ('a.wav', 'b.wav', 'o.wav', 'q.wav')); sf.write(pa, wet, SR, subtype='FLOAT'); sf.write(pb, dryb, SR, subtype='FLOAT')
    M = run(exe, ['both', pa, pb, po, pq]); rd = lambda p: sf.read(p, dtype='float64', always_2d=True)[0][:n]
    row = dict(i=i, mode=mode, effect=name, fam=fam, par=par, seconds=seconds, sound=kind, soundb=kindb, clips=used, clipsb=usedb, **brief(M))
    if M['found']:
        out = rd(po); row['out'] = score.removed(dry, wet, out)[0]; row['out_blocks'] = blocks(dry, wet, out, n, live)
        row['out_pieces'] = [score.removed(dry[p['a']:p['b']], wet[p['a']:p['b']], out[p['a']:p['b']])[0] if p['b'] - p['a'] >= 3 * SR else None for p in M['pieces']]
        row['live_pieces'] = [1.0 if live is None else float(live[p['a']:p['b']].mean()) for p in M['pieces']]
        row['win_live'] = [1.0 if live is None else float(live[w[0]:w[1]].mean()) for w in M.get('windows', [])]
        row['there'] = float(sum(b - a for a, b in M.get('spans', [])) / n)
        if live is not None: row['out_off'] = float(10 * np.log10((((out - wet) * (1 - live[:, None])) ** 2).sum() / max(((dry * (1 - live[:, None])) ** 2).sum(), 1e-20) + 1e-20))      # what was done to the half without the effect, dB against it
        out = rd(pq); row['onto'] = score.removed(wetb, dryb, out)[0]; row['onto_blocks'] = blocks(wetb, dryb, out, n, live)
    else: row['out'] = 0.0; row['onto'] = 0.0
    for p in (pa, pb, po, pq):
        if os.path.exists(p): os.remove(p)
    os.rmdir(d); return row

WANT = dict(tremolo='together', autopan='against', gatedrhythm='together')
def right(r): return r['found'] and r['score'] >= 0 and r['kind'] == WANT[r['fam']]
def show(paths, detail=True):
    for path in paths:
        rows = sorted((json.loads(l) for l in open(path)), key=lambda r: r['i'])
        if not rows: continue
        if 'effect' not in rows[0]:
            print('LONG CLEAN SOUNDS (%s): %d, a movement shown at the default Sureness on %d, at the lowest Sureness on %d; mean analysis time %.1f s' % (path, len(rows), sum(r['found'] and r['score'] >= 0 for r in rows), sum(r['found'] for r in rows), np.mean([r['took'] for r in rows])))
            for r in rows:
                if r['found']: print('    reported: #%d %d s %s %s%s score %.2f (sure %.2f) rate %.3f Hz depth %.1f dB, %d of %d pieces' % (r['i'], r['seconds'], r['sound'], r['kind'], ' (long look)' if r.get('wide') else '', r['score'], r['sure'], r['rate'], r['depth'], r['carrying'], r['sounding']))
        else:
            print('LONG SOUNDS, %s (%s): %d' % (rows[0].get('mode', 'wet'), path, len(rows)))
            for fam in ('tremolo', 'autopan', 'gatedrhythm', 'all'):
                rs = [r for r in rows if fam == 'all' or r['fam'] == fam]
                if not rs: continue
                ok = [r for r in rs if right(r)]; out = np.array([r['out'] if right(r) else 0.0 for r in rs]); onto = np.array([r['onto'] if right(r) else 0.0 for r in rs])
                fo = np.array([r['out'] for r in ok]) if ok else np.zeros(1); fn = np.array([r['onto'] for r in ok]) if ok else np.zeros(1)
                bl = np.array([v for r in ok for v in r.get('out_blocks', [])]) if ok else np.zeros(1); wrong = sum(r['found'] and r['score'] >= 0 and not right(r) for r in rs)
                print('  %-12s: shown %2d/%-2d (other kind shown %d) | out: median %5.1f mean %5.1f, where shown %5.1f / %5.1f, worse than nothing %d, stretches of 6 s under 50: %2.0f%% | onto: median %5.1f mean %5.1f, where shown %5.1f / %5.1f, worse %d | pieces carrying it %.0f%% | %.1f s'
                      % (fam, len(ok), len(rs), wrong, np.median(out), out.mean(), np.median(fo), fo.mean(), int((out < 0).sum()), 100 * (bl < 50).mean(), np.median(onto), onto.mean(), np.median(fn), fn.mean(), int((onto < 0).sum()),
                         100 * (np.mean([r['carrying'] / max(1, r['sounding']) for r in ok]) if ok else 0), np.mean([r['took'] for r in rs])))
            if not detail: continue
            for r in rows:
                print('    #%-3d %-17s %2d s %-15s %-9s score %s rate %6.3f Hz (%s) depth %5.1f dB pieces %d/%d%s | out %6.1f %s%s | onto %6.1f %s' % (r['i'], r['effect'], r['seconds'], r['sound'], r['kind'] if r['found'] else 'nothing', '%5.2f' % r['score'] if r['found'] else '    -', r['rate'],
                      ' '.join('%s=%.3g' % (k, v) for k, v in r['par'].items() if k in ('speed', 'hz', 'freq', 'rate', 'auto')), r['depth'], r['carrying'], r['sounding'], ' long look' if r.get('wide') else '', r['out'], ' '.join('%.0f' % v for v in r.get('out_blocks', [])),
                      ' (off half %.0f dB)' % r['out_off'] if 'out_off' in r else '', r['onto'], ' '.join('%.0f' % v for v in r.get('onto_blocks', []))))

if __name__ == '__main__':
    if sys.argv[1] == '--show': show([a for a in sys.argv[2:] if a != '--brief'], '--brief' not in sys.argv); sys.exit(0)
    exe, mode = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/claude-0/v2/level/long_%s.jsonl' % mode
    done = {json.loads(l)['i'] for l in open(out)} if os.path.exists(out) else set()
    if mode == 'clean': jobs = [(exe, i, LENGTHS[i % 4], (i // 4) % 2 == 0) for i in range(48)]; fn = clean_case
    elif mode == 'wet': jobs = [(exe, mode, 8 * e + j, name, LENGTHS[j % 4], j < 4) for e, name in enumerate(EFFECTS) for j in range(8)]; fn = wet_case
    elif mode == 'part': jobs = [(exe, mode, 4 * e + j, name, LENGTHS[1 + j % 3], j < 2) for e, name in enumerate(EFFECTS) for j in range(4)]; fn = wet_case
    elif mode == 'drift': jobs = [(exe, mode, 8 * e + j, name, LENGTHS[j % 4], j < 4) for e, name in enumerate(('trem', 'pan')) for j in range(8)]; fn = wet_case
    elif mode == 'slow': jobs = [(exe, mode, 8 * e + j, name, LENGTHS[j % 4], j < 4) for e, name in enumerate(('pan_sox', 'pan_tap', 'pan_calf')) for j in range(8)]; fn = wet_case
    jobs = [j for j in jobs if j[2 if mode != 'clean' else 1] not in done]
    with mp.Pool(int(os.environ.get('JOBS', '3'))) as pool:
        for row in pool.imap_unordered(fn, jobs):
            open(out, 'a').write(json.dumps(row) + '\n'); print('#%d %s %ds found %s' % (row['i'], row.get('effect', 'clean'), row['seconds'], row['found']), flush=True)
    show([out])
