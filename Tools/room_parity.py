"""Parity of the C++ room engine (Source/Core/Room.h, RoomNumbers.h, RoomEngine.h, run through the drroom tool) with
the Python prototype.

  python room_parity.py PROTOTYPE_DIR MODEL DRROOM SOUND [SOUND ...] [--second MODEL2] [--tmp DIR] [--keep] [-v]

PROTOTYPE_DIR holds common.py, net.py, render.py, roomnum.py, roomcap.py, predict.py and engine.py; MODEL is what
the trainer saves (.pt: it is copied first and exported with export_room_model.py) or a file exported earlier
(.bin, e.g. Resources/room.bin: Python's side then runs on the weights read back from it); DRROOM is the built
tool; SOUND are 48 kHz files (give at least 6). The second sound of "the found room on a second sound" is the next
in the list. With --second the table is for the two models in a row, as the app runs them when it has both: the
second one (7 input planes) looks again with the first one's answer.

For every sound the tool and Python compute the same things, and the table gives the largest and the average
difference, step by step:
  pictures      the level picture LE and the unshared shares U64, U256 (dB), the model's input planes;
  gains         the model's answer (dB): each side from its own pictures, and Python's model on the tool's input;
  numbers       the room in numbers: strength (dB), decay time (%), evidence, width per band. "End to end": each side
                from its own gains. "Same gains": Python's reader on the tool's gains, the reader alone. (The reader
                takes yes/no decisions at fixed levels, so a gain that differs in the fifth decimal can move a number:
                where end to end is off and same gains is not, that is what happened, and -v says on which sound);
  sounds        the impulse response, the found room on a second sound (onto) and the sound itself with its room
                turned down / up and reshaped (one_sound), in dB relative to the result: "same numbers" = Python's
                engine fed with the tool's gains and numbers, "end to end" = Python all the way. one_sound runs at
                amounts 0, 0.5, 1 and 2 with every fader setting, onto with the moved faders at 0.5 and 2.
Targets: pictures within 0.01 dB where the level is above -100 dB re the loudest cell, gains within 0.02 dB, found
or not the same on every sound, strength within 0.05 dB, decay time within 0.5 %, width within 0.005, sounds at
least 80 dB below the signal.
After the table: the first form of the tool must give what --one gives with the faders in the middle, the model
run in short pieces must give the same answer as in one go, damaged model files must be refused, models of other
sizes (random weights) must agree with Python, and so must a second pass (a random model of 7 input planes)."""
import importlib.util, os, shutil, struct, subprocess, sys, tempfile
import numpy as np, soundfile as sf, torch
sys.dont_write_bytecode = True                           # no cache folders next to the scripts

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]; keep = '--keep' in sys.argv; verbose = '-v' in sys.argv
    tmp = None; second = None
    if '--tmp' in sys.argv: tmp = sys.argv[sys.argv.index('--tmp') + 1]; args.remove(tmp)
    if '--second' in sys.argv: second = sys.argv[sys.argv.index('--second') + 1]; args.remove(second)
    if len(args) < 4: sys.exit(__doc__)
    proto, ckpt, tool, sounds = args[0], args[1], os.path.abspath(args[2]), args[3:]
    made = tmp is None; tmp = tempfile.mkdtemp() if made else tmp; os.makedirs(tmp, exist_ok=True); P = lambda name: os.path.join(tmp, name)
    def module(name, folder):                           # from that folder and no other (the prototype's files name a folder of their own)
        spec = importlib.util.spec_from_file_location(name, os.path.join(folder, name + '.py')); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
    ex = module('export_room_model', os.path.dirname(os.path.abspath(__file__)))
    cm, net, render, roomnum, roomcap, predict, engine = [module(n, proto) for n in ('common', 'net', 'render', 'roomnum', 'roomcap', 'predict', 'engine')]
    torch.set_num_threads(1); B = cm.NB
    def loaded(path, name):
        """A model for both sides: the exported file `name`.bin for the tool and the weights for Python."""
        if path.endswith('.bin'):
            shutil.copyfile(path, P(name + '.bin')); ck = ex.read(P(name + '.bin')); ck['state'] = {k: torch.from_numpy(v) for k, v in ck['state'].items()}
            if ck['deepest'] != net.GMIN: sys.exit('%s: its answers run down to -%g dB, the prototype\'s to -%g' % (path, ck['deepest'], net.GMIN))
        else: shutil.copyfile(path, P(name + '.pt')); ck = torch.load(P(name + '.pt'), map_location='cpu'); ex.export(ck, P(name + '.bin'), net.GMIN)
        m = net.UNet(**ck['cfg']); m.load_state_dict(ck['state']); m.eval(); return ck, m
    ck, model = loaded(ckpt, 'room'); model2 = None; SEC = ()
    if second: model2 = loaded(second, 'room2')[1]; SEC = ('second=' + P('room2.bin'),)

    def run(*a, fail=True):
        r = subprocess.run([tool] + list(a), capture_output=True, text=True)
        if fail and r.returncode != 0: sys.exit('the tool failed: %s\n%s' % (' '.join(a), r.stderr))
        return r.stdout.strip() if fail else r
    def raw(name, shape, kind=np.float32): return np.fromfile(P('d.%s' % name), kind).reshape(shape)
    def dumped(T, planes=4):
        num = raw('NUM.f64', (-1,), np.float64); c = dict(LE=raw('LE.f32', (T, B)), U64=raw('U64.f32', (T, B)), U256=raw('U256.f32', (T, B)), IN=raw('IN.f32', (planes, T, B)), G=raw('G.f32', (T, B)), found=bool(num[0]))
        c['kdb'], c['T60'], c['ev'], c['width'] = num[1:].reshape(4, B); return c
    def answer(m, x):
        with torch.no_grad(): return m(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None])[0].numpy()
    def planes_of(ft): return net.inputs(ft['LE'].astype(np.float32), ft['U64'].astype(np.float32), ft['U256'].astype(np.float32))   # single precision first, as the prototype's door to the model does
    def answers(ft):
        """Python's input planes and answer: the first model's, and with a second model its planes (the first four and the three made from the first answer) and its answer; the first answer."""
        planes = planes_of(ft); g1 = answer(model, planes)
        if model2 is None: return planes, g1, g1
        planes = np.concatenate([planes, roomnum.second_look(ft['LE'], g1.astype(np.float64))]); return planes, answer(model2, planes), g1
    def analysed(x, g):
        """The prototype's analyse() with these gains in the place of its own model's."""
        predict.gains = lambda ft, name=None: np.asarray(g, np.float64); return engine.analyse(x)
    def load(path, name):
        """The sound as both sides will read it: written as a WAV the tool understands, and read back."""
        x, sr = sf.read(path, always_2d=True)
        if sr != cm.SR: sys.exit('%s is not at 48 kHz' % path)
        x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]
        sf.write(P(name), x, sr, subtype='PCM_24' if sf.info(path).subtype in ('PCM_16', 'PCM_24', 'PCM_S8', 'PCM_U8') else 'FLOAT')
        return sf.read(P(name), always_2d=True)[0]
    def apart(y, r):
        """Difference of two sounds in dB re the second: of the whole (rms), of the worst sample (re the peak)."""
        if y.shape != r.shape: return 999.0, 999.0
        e = y - r; return 10 * np.log10(max((e ** 2).sum(), 1e-300) / max((r ** 2).sum(), 1e-300)), 20 * np.log10(max(np.abs(e).max(), 1e-150) / max(np.abs(r).max(), 1e-150))
    def numbers_apart(a, t):
        """Strength (dB), decay time (%), evidence, width: the differences per band between two readings that both found a room."""
        return dict(kdb=np.abs(a['kdb'] - t['kdb']), T60=100 * np.abs(a['T60'] / t['T60'] - 1), ev=np.abs(a['ev'] - t['ev']), width=np.abs(a['width'] - t['width']))

    MIDDLE = (1.0, 1.0, 1.0); MOVED = ((0.5, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 0.6, 1.0), (1.0, 1.4, 1.0), (1.0, 1.0, 0.5), (1.0, 1.0, 1.5))       # length, brightness, width
    AMOUNTS = (0.0, 0.5, 1.0, 2.0)
    CASES = [('one_sound', a, f) for f in (MIDDLE,) + MOVED for a in AMOUNTS] + [('onto', a, MIDDLE) for a in AMOUNTS] + [('onto', a, f) for f in MOVED for a in (0.5, 2.0)]
    def label(f): return 'faders in the middle' if f == MIDDLE else ', '.join('%s %g' % (n, v) for n, v in zip(('length', 'brightness', 'width'), f) if v != 1.0)
    feat = {}; gain = {}; nums = {}; snd = {}; times = {}; flags = []; notes = []
    def add(d, key, v): d.setdefault(key, []).append(v)
    def timed(out):
        for key in ('find', 'impulse response', 'onto', 'one sound', 'render'):
            if ' ' + key + ' ' in out: add(times, key, float(out.split(' ' + key + ' ')[1].split('ms')[0]))
        return out
    for si, path in enumerate(sounds):
        x = load(path, 'in.wav'); b = load(sounds[(si + 1) % len(sounds)], 'b.wav'); n = len(x); T = n // cm.HOP + 6
        timed(run(P('room.bin'), P('in.wav'), '--dump', P('d'), *SEC)); c = dumped(T, 4 if model2 is None else 7)
        ft = cm.features(x); live = ft['LE'] > ft['LE'].max() - 100.0; planes, g, g1 = answers(ft)
        for k in ('LE', 'U64', 'U256'):
            add(feat, '%s (dB), cells above -100 dB' % k, np.abs(c[k] - ft[k])[live]); add(feat, '%s (dB), all cells' % k, np.abs(c[k] - ft[k]).ravel())
        add(feat, 'the model\'s input planes (-1.75 .. 1.75)', np.abs(c['IN'] - planes).ravel())
        if model2 is not None: add(gain, 'gains of the first model (dB)', np.abs(raw('G1.f32', (T, B)) - g1).ravel())
        add(gain, 'gains (dB)', np.abs(c['G'] - g).ravel()); add(gain, 'gains (dB), Python\'s model on the tool\'s input (model alone)', np.abs(c['G'] - answer(model if model2 is None else model2, c['IN'])).ravel())
        own = analysed(x, g); same = analysed(x, c['G']); flags.append((own['found'], same['found'], c['found']))
        line = '%2d  %5.1f s | gains %.6f dB | room: Python %s, Python on the tool\'s gains %s, tool %s' % (si + 1, n / cm.SR, np.abs(c['G'] - g).max(), own['found'], same['found'], c['found'])
        for ref, an in (('end to end', own), ('same gains', same)):
            if not (an['found'] and c['found']): continue
            d = numbers_apart(an, c)
            for k in d: add(nums, (k, ref), d[k])
            line += ' | %s: strength %.2g dB, decay time %.2g %%, width %.2g' % (ref, d['kdb'].max(), d['T60'].max(), d['width'].max())
            if ref == 'end to end' and (d['kdb'].max() > 0.05 or d['T60'].max() > 0.5 or d['width'].max() > 0.005):
                bad = np.nonzero((d['kdb'] > 0.05) | (d['T60'] > 0.5) | (d['width'] > 0.005))[0]
                notes.append('sound %d: end to end %d bands are off (bands %s): strength by up to %.3f dB, decay time by up to %.2f %%, width by up to %.4f' % (si + 1, len(bad), ' '.join(map(str, bad)), d['kdb'].max(), d['T60'].max(), d['width'].max()))
        if c['found']:
            tl = dict(LE=ft['LE'], g=c['G'].astype(np.float64), found=True, kdb=c['kdb'], T60=c['T60'], ev=c['ev'], width=c['width'])       # what the tool found, for Python's engine
            for f in (MIDDLE,) + MOVED:
                timed(run(P('room.bin'), P('in.wav'), '--ir', P('out.wav'), *map(repr, f), *SEC)); add(snd, ('impulse response', label(f), 'same numbers'), apart(sf.read(P('out.wav'), always_2d=True)[0], engine.room_ir(tl, *f)))
            for what, a, f in CASES:
                fn, src, extra = (engine.onto, b, ('--onto', P('b.wav'), P('out.wav'), repr(a)) + (tuple(map(repr, f)) if f != MIDDLE else ())) if what == 'onto' else (engine.one_sound, x, ('--one', P('out.wav'), repr(a)) + tuple(map(repr, f)))
                timed(run(P('room.bin'), P('in.wav'), *extra, *SEC)); y = sf.read(P('out.wav'), always_2d=True)[0]; case = 'amount %g' % a if f == MIDDLE else label(f)
                add(snd, (what, case, 'same numbers'), apart(y, fn(src, tl, a, *f)))
                if own['found']: add(snd, (what, case, 'end to end'), apart(y, fn(src, own, a, *f)))
            line += ' | impulse response %.1f dB, onto %.1f dB, one_sound %.1f dB' % tuple(max(v[-1][0] for k, v in snd.items() if k[0] == w and k[2] == 'same numbers') for w in ('impulse response', 'onto', 'one_sound'))
        if verbose: print(line, flush=True)

    ok = lambda good: 'ok' if good else 'MISSED'
    print('\n%d sounds%s. Differences between the tool and Python:' % (len(sounds), '' if model2 is None else ', two models in a row'))
    print('%-72s %12s %12s   %s' % ('', 'largest', 'average', 'target'))
    for k, v in feat.items():
        v = np.concatenate(v); print('%-72s %12.2g %12.2g   %s' % (k, v.max(), v.mean(), '0.01  ' + ok(v.max() <= 0.01) if 'above' in k else 'none'))
    for k, v in gain.items():
        v = np.concatenate(v); print('%-72s %12.6f %12.6f   %s' % (k, v.max(), v.mean(), '0.02  ' + ok(v.max() <= 0.02)))
    agree = sum(1 for o, s, t in flags if o == t); print('%-72s %12s %12s   %s' % ('a room found or not: the same on', '%d of %d' % (agree, len(flags)), '', 'all   ' + ok(agree == len(flags))) + '   (found by the tool on %d)' % sum(1 for f in flags if f[2]))
    LIMIT = dict(kdb=0.05, T60=0.5, width=0.005); NAME = dict(kdb='strength (dB)', T60='decay time (%)', ev='evidence (moments)', width='width (0 .. 1)')
    for ref in ('end to end', 'same gains'):
        for k in ('kdb', 'T60', 'ev', 'width'):
            if (k, ref) not in nums: continue
            v = np.concatenate(nums[(k, ref)]); print('%-72s %12.2g %12.2g   %s' % ('%s, %s' % (NAME[k], ref), v.max(), v.mean(), '%-5g ' % LIMIT[k] + ok(v.max() <= LIMIT[k]) if k in LIMIT else 'none'))
    print('%-72s %12s %12s   %s' % ('sounds, dB re the result', 'worst sound', 'average', 'target       (worst single sample re peak)'))
    for (what, case, ref), v in snd.items():
        v = np.array(v); print('%-72s %12.1f %12.1f   %-12s (%.1f)' % ('%s, %s, %s' % (what, case, ref), v[:, 0].max(), v[:, 0].mean(), '-80   ' + ok(v[:, 0].max() <= -80.0), v[:, 1].max()))
    for nt in notes: print(nt)
    print('tool, mean time over these runs: ' + ', '.join('%s %.0f ms' % (k, np.mean(v)) for k, v in times.items()))

    # ---- the first form of the tool is --one with the faders in the middle
    x = load(sounds[0], 'in.wav'); T = len(x) // cm.HOP + 6; ft = cm.features(x)
    run(P('room.bin'), P('in.wav'), P('out.wav'), '0.5'); a = open(P('out.wav'), 'rb').read(); run(P('room.bin'), P('in.wav'), '--one', P('out.wav'), '0.5', '1', '1', '1')
    print('drroom model in out 0.5 against --one out 0.5 1 1 1: %s' % ('the same file  ok' if a == open(P('out.wav'), 'rb').read() else 'DIFFERENT'))

    # ---- the model in short pieces: the same answer as in one go
    step = 2 ** (len(ck['cfg']['ch']) - 1); pieces = (step, 3 * step, 10 * step)
    run(P('room.bin'), P('in.wav'), '--dump', P('d')); whole = dumped(T); worst = 0.0
    for piece in pieces:
        run(P('room.bin'), P('in.wav'), '--dump', P('d'), 'piece=%d' % piece); worst = max(worst, float(np.abs(raw('G.f32', (T, B)) - whole['G']).max()))
    print('model run in pieces of %d, %d and %d frames against one go: largest difference %g dB  %s' % (pieces + (worst, ok(worst == 0.0))))

    # ---- damaged model files are refused (exit code 1 and a message, no crash)
    good = open(P('room.bin'), 'rb').read(); L = len(ck['cfg']['ch']); head = 20 + 4 * L; bad = {}
    def put(blob, at, fmt, v): return blob[:at] + struct.pack(fmt, v) + blob[at + struct.calcsize(fmt):]
    bad['empty file'] = b''; bad['only the magic'] = good[:4]; bad['wrong magic'] = b'XXXX' + good[4:]; bad['version 2'] = put(good, 4, '<I', 2); bad['five input planes'] = put(good, 8, '<I', 5)
    bad['seven input planes, layers for four'] = put(good, 8, '<I', 7)
    bad['gain range not a number'] = put(good, 12, '<f', float('nan')); bad['no levels'] = put(good, 16, '<I', 0); bad['a thousand levels'] = put(good, 16, '<I', 1000)
    bad['a level with no channels'] = put(good, 20, '<I', 0); bad['a level with too many channels'] = put(good, 20, '<I', 0xFFFFFFFF); bad['another channel count than the layers have'] = put(good, 20, '<I', ck['cfg']['ch'][0] + 1)
    bad['a layer of the wrong size'] = put(good, head, '<I', 7); bad['cut inside the header'] = good[:head - 2]; bad['cut inside a layer'] = good[:len(good) // 2]; bad['last byte missing'] = good[:-1]
    bad['one byte too many'] = good + b'\0'; bad['a weight that is not a number'] = put(good, head + 4 + 40, '<f', float('nan')); bad['a bias that is infinite'] = put(good, len(good) - 4, '<f', float('inf'))
    fails = []
    for what, blob in bad.items():
        open(P('bad.bin'), 'wb').write(blob)
        for extra in ((P('bad.bin'), P('in.wav'), '--dump', P('d')), (P('room.bin'), P('in.wav'), '--dump', P('d'), 'second=' + P('bad.bin'))):      # as the model, and as the second model
            r = run(*extra, fail=False)
            if r.returncode != 1 or 'room model:' not in r.stderr: fails.append('%s (exit code %d: %s)' % (what, r.returncode, r.stderr.strip()))
            elif verbose and extra[0] == P('bad.bin'): print('  %-44s -> %s' % (what, r.stderr.strip()))
    r = run(P('room.bin'), P('in.wav'), '--dump', P('d'), 'second=' + P('room.bin'), fail=False)            # a model of 4 planes cannot be the second pass
    if r.returncode != 1 or 'room model:' not in r.stderr: fails.append('a first-pass model given as the second (exit code %d: %s)' % (r.returncode, r.stderr.strip()))
    print('%d damaged model files, each as the model and as the second model: %s' % (len(bad), 'all refused  ok' if not fails else 'NOT refused: ' + '; '.join(fails)))

    # ---- models of other sizes, random weights: the tool against Python on the tool's input planes
    def spread_out(m, planes):                                   # the last layer is set so that the answers spread over the whole range
        with torch.no_grad():
            z = answer(m, planes).astype(np.float64) / -net.GMIN; z = np.log(z / (1.0 - z)); k = 1.5 / z.std(); b = float(m.head.bias)
            m.head.weight.mul_(k); m.head.bias.fill_(-k * (z.mean() - b))
    torch.manual_seed(1); worst = 0.0; shapes = ((5,), (6, 9), (8, 8, 16), (16, 12, 8), (4, 6, 8, 10, 12, 14, 16))
    for ch in shapes:
        m = net.UNet(ch=ch); m.eval(); spread_out(m, whole['IN']); ex.export(dict(cfg=m.cfg, state=m.state_dict()), P('other.bin'), net.GMIN)
        for piece in (4096, 2 ** (len(ch) - 1)):
            run(P('other.bin'), P('in.wav'), '--dump', P('d'), 'piece=%d' % piece); g = raw('G.f32', (T, B)); d = float(np.abs(g - answer(m, raw('IN.f32', (4, T, B)))).max()); worst = max(worst, d)
            if verbose: print('  levels %-28s in pieces of %4d frames: gains from %.1f to %.1f dB, largest difference %.6f dB' % (ch, piece, g.min(), g.max(), d))
    print('models with levels %s: largest gain difference %.6f dB  %s' % (', '.join(str(list(c)) for c in shapes), worst, ok(worst <= 0.02)))

    # ---- a second pass: a random model of 7 input planes behind the real one
    g1 = answer(model, planes_of(ft)); look = roomnum.second_look(ft['LE'], g1.astype(np.float64)); m2 = net.UNet(cin=7); m2.eval(); spread_out(m2, np.concatenate([planes_of(ft), look]))
    ex.export(dict(cfg=m2.cfg, state=m2.state_dict()), P('second.bin'), net.GMIN); g2 = answer(m2, np.concatenate([planes_of(ft), look]))
    for piece in (4096, step):
        run(P('room.bin'), P('in.wav'), '--dump', P('d'), 'second=' + P('second.bin'), 'piece=%d' % piece); c = dumped(T, 7); t1 = raw('G1.f32', (T, B)); same = analysed(x, c['G'])
        d = (np.abs(c['IN'][4:] - roomnum.second_look(ft['LE'], t1.astype(np.float64))).max(), np.abs(c['G'] - answer(m2, c['IN'])).max(), np.abs(c['IN'][4:] - look).max(), np.abs(c['G'] - g2).max())
        nd = numbers_apart(same, c) if same['found'] and c['found'] else None
        print('second pass, pieces of %d frames: the three extra planes from the same first answer %.2g (planes run -1 .. 1.5; a room was %s), the second model alone %.6f dB | end to end: planes %.2g, gains %.6f dB (from %.1f to %.1f)  %s' % (
            piece, d[0], 'found' if look.max() > -1 else 'not found', d[1], d[2], d[3], c['G'].min(), c['G'].max(), ok(d[0] <= 1e-5 and d[1] <= 0.02 and d[3] <= 0.02)))
        print('    numbers read from the second answer, same gains: %s' % ('room %s / %s' % (same['found'], c['found']) if nd is None else 'strength %.2g dB, decay time %.2g %%, width %.2g  %s' % (nd['kdb'].max(), nd['T60'].max(), nd['width'].max(), ok(nd['kdb'].max() <= 0.05 and nd['T60'].max() <= 0.5 and nd['width'].max() <= 0.005))))
    if made and not keep: shutil.rmtree(tmp)
    elif not keep:
        for f in ('room.pt', 'room2.pt', 'room.bin', 'room2.bin', 'other.bin', 'second.bin', 'bad.bin', 'in.wav', 'b.wav', 'out.wav', 'd.LE.f32', 'd.U64.f32', 'd.U256.f32', 'd.IN.f32', 'd.G.f32', 'd.G1.f32', 'd.NUM.f64'):
            if os.path.exists(P(f)): os.remove(P(f))

if __name__ == '__main__': main()
