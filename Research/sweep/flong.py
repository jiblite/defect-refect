# The flanger engine on long sounds, as the app uses it (find / apply in Source/Core/Sweep.h, through Tests/sweep.cpp =
# drsweep both). Long sounds are joined 6 s corpus clips (see Research/level/long.py).
#   python3 flong.py path/to/drsweep clean [out.jsonl]    long clean sounds: is a flanger reported? how long does it take?
#   python3 flong.py path/to/drsweep wet [out.jsonl]      long sounds through outside flangers: found? taken out? put on a second long sound?
#   python3 flong.py path/to/drsweep part [out.jsonl]     the flanger on one half of the sound only
#   python3 flong.py --show file.jsonl [...]
# SEED=1: a second draw. Real recordings are named by their corpus ids only.
import sys, os, json, subprocess, tempfile, multiprocessing as mp
import numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/claude-0/v2/level'); sys.path.insert(0, '/tmp/claude-0/v2/detect/py'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import long as L
SR = 48000; EFFECTS = ('flg_calf', 'flg_sox', 'flg_swh', 'flg_mda_thruzero'); LENGTHS = (33, 47, 62)

def run(exe, args):
    r = subprocess.run([exe] + args, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-300:] + r.stderr[-300:])
    return json.loads(r.stdout.strip().splitlines()[-1])
def raw(p, n): return np.fromfile(p, dtype='float64').reshape(2, -1).T[:n]
def clean_case(a):
    exe, i, seconds, same = a; rng = np.random.default_rng(L.SEED + 27000 + i); x, kind, used = L.long_sound(rng, seconds, same)
    d = tempfile.mkdtemp(dir='/dev/shm'); pa = os.path.join(d, 'a.wav'); sf.write(pa, x, SR, subtype='FLOAT'); M = run(exe, ['both', pa, pa, os.path.join(d, 'o.f64'), os.path.join(d, 'q.f64')])
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    os.rmdir(d); return dict(i=i, mode='clean', seconds=seconds, sound=kind, clips=used, **M)
def wet_case(a):
    import score, zoo_fx
    exe, mode, i, name, seconds, same = a; rng = np.random.default_rng(L.SEED + dict(wet=29000, part=31000)[mode] + i); n = int(seconds * SR)
    par = zoo_fx.ZOO[name][2](rng); live = None
    if mode == 'part':
        live = np.zeros(n); cut = int(n * rng.uniform(0.4, 0.6)); X = int(0.05 * SR)
        if rng.random() < 0.5: live[:cut] = 1; live[cut:cut + X] = np.linspace(1, 0, X)
        else: live[cut:] = 1; live[cut - X:cut] = np.linspace(0, 1, X)
        def fn(x): w = zoo_fx.apply(name, x, par); m = live[:len(x), None]; return x * (1 - m) + w * m
    else: fn = lambda x: zoo_fx.apply(name, x, par)
    dry, kind, used = L.long_sound(rng, seconds, same); dryb, kindb, usedb = L.long_sound(rng, seconds, same); wet = fn(dry); wetb = fn(dryb)
    d = tempfile.mkdtemp(dir='/dev/shm'); pa, pb, po, pq = (os.path.join(d, f) for f in ('a.wav', 'b.wav', 'o.f64', 'q.f64')); sf.write(pa, wet, SR, subtype='FLOAT'); sf.write(pb, dryb, SR, subtype='FLOAT')
    M = run(exe, ['both', pa, pb, po, pq])
    row = dict(i=i, mode=mode, effect=name, par={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in par.items()}, seconds=seconds, sound=kind, soundb=kindb, clips=used, clipsb=usedb, **M)
    if M['found']:
        out = raw(po, n); row['out'] = score.removed(dry, wet, out)[0]; row['out_blocks'] = L.blocks(dry, wet, out, n, live)
        if live is not None: row['out_off'] = float(10 * np.log10((((out - wet) * (1 - live[:, None])) ** 2).sum() / max(((dry * (1 - live[:, None])) ** 2).sum(), 1e-20) + 1e-20))
        row['live_parts'] = [1.0 if live is None else float(live[p['a']:p['b']].mean()) for p in M['parts']]
        out = raw(pq, n); row['onto'] = score.removed(wetb, dryb, out)[0]; row['onto_blocks'] = L.blocks(wetb, dryb, out, n, live)
    else: row['out'] = 0.0; row['onto'] = 0.0
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    os.rmdir(d); return row
def shown(r): return r['found'] and (r['sounding'] < 2 or r['shown'] >= 2)       # at the default Sureness
def show(paths):
    for path in paths:
        rows = sorted((json.loads(l) for l in open(path)), key=lambda r: r['i'])
        if not rows: continue
        if 'effect' not in rows[0]:
            print('LONG CLEAN SOUNDS (%s): %d, a flanger shown at the default Sureness on %d, at a lower one on %d; analysis %.1f s a sound, %.2f s for every second of sound' % (path, len(rows), sum(shown(r) for r in rows), sum(r['found'] for r in rows), np.mean([r['took'] for r in rows]), np.sum([r['took'] for r in rows]) / np.sum([r['seconds'] for r in rows])))
            for r in rows:
                if r['found']: print('    reported: #%d %d s %s  %d of %d pieces, gain %.2f, delay %.1f .. %.1f samples, %s' % (r['i'], r['seconds'], r['sound'], r['shown'], r['sounding'], r['g'], r['dmin'], r['dmax'], ('%.2f Hz' % r['rate']) if r['periodic'] else 'free'))
        else:
            print('LONG SOUNDS, %s (%s): %d' % (rows[0]['mode'], path, len(rows)))
            for eff in EFFECTS + ('all',):
                rs = [r for r in rows if eff == 'all' or r['effect'] == eff]
                if not rs: continue
                out = np.array([r['out'] if shown(r) else 0.0 for r in rs]); onto = np.array([r['onto'] if shown(r) else 0.0 for r in rs])
                print('  %-17s: shown %2d/%-2d (found at a lower Sureness %d) | out: median %5.1f mean %5.1f, worse than nothing %d | onto: median %5.1f mean %5.1f, worse %d | pieces showing it %.0f%% | %.1f s'
                      % (eff, sum(shown(r) for r in rs), len(rs), sum(r['found'] and not shown(r) for r in rs), np.median(out), out.mean(), int((out < 0).sum()), np.median(onto), onto.mean(), int((onto < 0).sum()),
                         100 * np.mean([r['shown'] / max(1, r['sounding']) for r in rs]), np.mean([r['took'] for r in rs])))
            for r in rows:
                print('    #%-3d %-17s %2d s %-15s %s pieces %d/%d [%s] gain %5.2f delay %5.1f..%5.1f %-8s | out %6.1f %s%s | onto %6.1f %s' % (r['i'], r['effect'], r['seconds'], r['sound'], 'found  ' if r['found'] else 'nothing', r['shown'], r['sounding'],
                      ''.join('x' if p['found'] else '.' for p in r['parts']), r['g'], r['dmin'], r['dmax'], ('%.2f Hz' % r['rate']) if r['periodic'] else 'free', r['out'], ' '.join('%.0f' % v for v in r.get('out_blocks', [])),
                      ' (off half %.0f dB)' % r['out_off'] if 'out_off' in r else '', r['onto'], ' '.join('%.0f' % v for v in r.get('onto_blocks', []))))
if __name__ == '__main__':
    if sys.argv[1] == '--show': show(sys.argv[2:]); sys.exit(0)
    exe, mode = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/claude-0/v2/sweep/flong_%s.jsonl' % mode
    done = {json.loads(l)['i'] for l in open(out)} if os.path.exists(out) else set()
    if mode == 'clean': jobs = [(exe, i, LENGTHS[i % 3], (i // 3) % 2 == 0) for i in range(24)]; fn = clean_case
    else: jobs = [(exe, mode, 6 * e + j, name, LENGTHS[j % 3], j < 3) for e, name in enumerate(EFFECTS) for j in range(6)]; fn = wet_case
    jobs = [j for j in jobs if j[2 if mode != 'clean' else 1] not in done]
    with mp.Pool(int(os.environ.get('JOBS', '3'))) as pool:
        for row in pool.imap_unordered(fn, jobs):
            open(out, 'a').write(json.dumps(row) + '\n'); print('#%d %s %ds found %s (%d/%d) %.1f s' % (row['i'], row.get('effect', 'clean'), row['seconds'], row['found'], row['shown'], row['sounding'], row['took']), flush=True)
    show([out])
