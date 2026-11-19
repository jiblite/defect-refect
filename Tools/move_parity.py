#!/usr/bin/env python3
"""Compares the app's MOVEMENT faders (Tests/move.cpp, built as drmove) with the Python prototype (Research/movement).

  python3 Tools/move_parity.py path/to/drmove a.wav b.wav|- [a2.wav b2.wav|- ...] [--faders 0,0,0,0,0 ...] [--cents made|proto|app|none] [--keep folder]

For each pair (48 kHz WAVs): with - in place of b the faders work on a alone (one sound); otherwise they work on b and a is
the other sound, the one whose movement the right half of a fader brings. --faders: Pitch, Loudness, Tone, Stereo, Spectral,
each 0 .. 2; several sets may be given. Both sides work out what moves in the sounds and put the faders on.
--cents: where the pitch curves come from.
   made   a made-up curve (a vibrato on a glide), the same for both sides: checks how a pitch curve is put on (default)
   proto  the prototype's reader, handed to the app as well: the same, with real curves
   app    each side reads the pitch movement by itself: checks the reader (slow on the prototype's side)
   none   no pitch movement
Curves are compared by their largest difference (dB, cents), sounds as the level of the difference against the sound (dB).
Needs numpy, scipy and soundfile.
"""
import sys, os, subprocess, tempfile, argparse
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser(); ap.add_argument('drmove'); ap.add_argument('wavs', nargs='+'); ap.add_argument('--faders', nargs='+', default=['0,0,0,0,0', '2,2,2,2,2', '0.5,1.5,0.5,1.5,0.5'])
ap.add_argument('--cents', default='made'); ap.add_argument('--keep', default=''); ap.add_argument('--proto', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Research', 'movement'))
A = ap.parse_args(); sys.path.insert(0, A.proto)
import move, flow

def rel_db(x, ref): return 10 * np.log10(max(float(((x - ref) ** 2).sum()), 1e-300) / max(float((ref ** 2).sum()), 1e-300))
def load(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p + ' is not at 48 kHz'
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def made(n, k):
    t = np.arange(1 + n // move.STEP) / 100.0; return (40.0 + 20 * k) * np.sin(2 * np.pi * (4.0 + k) * t) + np.linspace(-150, 150, len(t)) * (1 - 2 * k)
def reader(x):
    c, held = flow.track(x.mean(1))[:2]; return np.asarray(c, float), np.asarray(held) > 0
def found(pre, T5, T10):
    v = np.fromfile(pre + '.loud.f64'); L = dict(db=v[:T5], ref=float(v[T5])); v = np.fromfile(pre + '.space.f64')
    S = dict(pan=v[:T10], width=v[T10:2 * T10], power=v[2 * T10:3 * T10], pan0=float(v[3 * T10]), width0=float(v[3 * T10 + 1]), mono=bool(v[3 * T10 + 2]))
    v = np.fromfile(pre + '.tone.f64'); M = dict(m=v[:T10 * move.NBAND].reshape(T10, move.NBAND), w=v[T10 * move.NBAND:]); v = np.fromfile(pre + '.pitch.f64')
    return dict(loud=L, space=S, tone=M, spec=np.fromfile(pre + '.spec.f64'), pitch=dict(cents=v[:len(v) // 2], held=v[len(v) // 2:] > 0))
def differences(F, G):
    """F: the prototype's, G: the app's."""
    d = {}; on = F['loud']['db'] > -190
    d['level dB'] = max(float(np.abs(F['loud']['db'] - G['loud']['db'])[on].max()) if on.any() else 0.0, abs(F['loud']['ref'] - G['loud']['ref']))
    d['pan dB'] = max(float(np.abs(F['space']['pan'] - G['space']['pan']).max()), abs(F['space']['pan0'] - G['space']['pan0']))
    d['width dB'] = max(float(np.abs(F['space']['width'] - G['space']['width']).max()), abs(F['space']['width0'] - G['space']['width0']))
    d['tone dB'] = max(float(np.abs(F['tone']['m'] - G['tone']['m']).max()), float(np.abs(F['tone']['w'] - G['tone']['w']).max()))
    on = F['spec'] > -200; d['colour dB'] = float(np.abs(F['spec'] - G['spec'])[on].max()) if on.any() else 0.0
    if F['space']['mono'] != G['space']['mono']: d['mono or not'] = 1.0
    return d

worst = {}; worst_sound = -999.0
for a_path, b_path in zip(A.wavs[0::2], A.wavs[1::2]):
    two = b_path != '-'; a = load(a_path); b = load(b_path) if two else None; x = b if two else a
    out = A.keep or tempfile.mkdtemp(); pre = os.path.join(out, os.path.splitext(os.path.basename(a_path))[0] + '__' + (os.path.splitext(os.path.basename(b_path))[0] if two else 'alone'))
    own = move.find(x); other = move.find(a) if two else None; files = ['-', '-']
    if A.cents in ('made', 'proto'):
        for k, (F, s) in enumerate(((own, x), (other, a))):
            if F is None: continue
            c = made(len(s), k) if A.cents == 'made' else reader(s)[0]; F['pitch'] = dict(cents=c, held=np.ones(len(c), bool)); files[k] = pre + '.cents%d.f64' % k; c.astype(np.float64).tofile(files[k])
    elif A.cents == 'app':
        for F, s in ((own, x), (other, a)):
            if F is not None: c, h = reader(s); F['pitch'] = dict(cents=c, held=h)
    print('%s | %s' % (os.path.basename(a_path), os.path.basename(b_path) if two else 'alone'))
    for fs in A.faders:
        v = [float(q) for q in fs.split(',')]; faders = dict(zip(('pitch', 'loudness', 'tone', 'stereo', 'spectral'), v))
        if A.cents == 'none': faders['pitch'] = 1.0; v[0] = 1.0
        r = subprocess.run([A.drmove, a_path, b_path, pre] + ['%r' % q for q in v] + files, capture_output=True, text=True)
        if r.returncode: print(r.stdout, r.stderr); sys.exit(1)
        T5 = 1 + len(x) // move.LSTEP; T10 = 1 + len(x) // move.STEP; G = found(pre + '.own', T5, T10); d = differences(own, G)
        if two: d2 = differences(other, found(pre + '.other', 1 + len(a) // move.LSTEP, 1 + len(a) // move.STEP)); d = {k: max(d[k], d2.get(k, 0.0)) for k in d}
        if A.cents == 'app':
            pc = own['pitch']; d['pitch cents'] = float(np.abs(pc['cents'] - G['pitch']['cents']).max()); d['followed or not'] = float((pc['held'] != G['pitch']['held']).mean())
            if two:                                 # the app's own reading of the other sound is used on the app's side
                Go = found(pre + '.other', 1 + len(a) // move.LSTEP, 1 + len(a) // move.STEP); d['pitch cents'] = max(d['pitch cents'], float(np.abs(other['pitch']['cents'] - Go['pitch']['cents']).max()))
        ref = move.apply(x, own, other, faders); y = np.fromfile(pre + '.out.f64').reshape(-1, 2); s = rel_db(y, ref) if float((ref ** 2).sum()) > 0 else -999.0
        for k in d: worst[k] = max(worst.get(k, 0.0), d[k])
        worst_sound = max(worst_sound, s)
        print('   faders %-22s found: %s | the result against the prototype\'s: %.0f dB' % (fs, ', '.join('%s %.1e' % kv for kv in d.items()), s))
print('worst differences: ' + ', '.join('%s %.1e' % kv for kv in worst.items()) + '; worst sound difference %.0f dB' % worst_sound)
