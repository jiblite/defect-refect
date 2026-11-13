# The effect zoo: effects from outside this project (plugins, sox, pedalboard, simulated rooms), with random but musical settings.
# Each entry: name -> (family, split, sampler(rng) -> params, apply(x, params) -> y).  x, y: (n, 2) at 48 kHz.
import numpy as np, scipy.signal as ss
from zoo_core import *
import pedalboard as pb

ZOO = {}
def reg(name, family, split, sampler, fn): ZOO[name] = (family, split, sampler, fn)
U = lambda rng, a, b: float(rng.uniform(a, b))
LU = lambda rng, a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))

# ------------------------------------------------------------------ reverb
reg('rev_dragonfly_hall', 'reverb', 'dev', lambda r: dict(dry_level=80, early_level=U(r, 0, 30), late_level=LU(r, 8, 100), size=U(r, 12, 50), decay=LU(r, 0.6, 5), diffuse=U(r, 50, 100), high_mult=U(r, 0.3, 0.9), low_mult=U(r, 0.8, 1.8), delay=U(r, 0, 40)),
    lambda x, p: run_lv2(x, 'https://github.com/michaelwillis/dragonfly-reverb', p))
reg('rev_dragonfly_room', 'reverb', 'dev', lambda r: dict(dry_level=80, early_level=U(r, 5, 40), late_level=LU(r, 12, 100), size=U(r, 8, 25), decay=LU(r, 0.2, 1.5), predelay=U(r, 0, 20), late_damp=LU(r, 3000, 14000)),
    lambda x, p: run_lv2(x, 'urn:dragonfly:room', p))
reg('rev_dragonfly_plate', 'reverb', 'test', lambda r: dict(dry_level=80, early_level=LU(r, 10, 100), algorithm=float(r.integers(0, 3)), decay=LU(r, 0.4, 4), predelay=U(r, 0, 30), low_cut=U(r, 20, 200), early_damp=LU(r, 4000, 15000)),
    lambda x, p: run_lv2(x, 'urn:dragonfly:plate', p))
reg('rev_calf', 'reverb', 'dev', lambda r: dict(decay_time=LU(r, 0.4, 5), hf_damp=LU(r, 2500, 15000), room_size=float(r.integers(0, 6)), diffusion=U(r, 0.3, 1), amount=LU(r, 0.15, 2.0), dry=1.0, predelay=U(r, 0, 40), bass_cut=LU(r, 40, 400), treble_cut=LU(r, 3000, 18000)),
    lambda x, p: run_lv2(x, 'http://calf.sourceforge.net/plugins/Reverb', p))
reg('rev_mda_ambience', 'reverb', 'dev', lambda r: dict(size=U(r, 0.2, 1), hf_damp=U(r, 0.2, 0.9), mix=U(r, 0.2, 0.9), output=0.5),
    lambda x, p: run_lv2(x, 'http://drobilla.net/plugins/mda/Ambience', p))
reg('rev_zita', 'reverb', 'test', lambda r: dict(delay=U(r, 0.02, 0.08), xover=LU(r, 100, 500), rtlow=LU(r, 1, 5), rtmid=LU(r, 1, 4), damping=LU(r, 2500, 12000), mix=U(r, 0.15, 0.85)),
    lambda x, p: run_ladspa(x, 'zita-reverbs.so', 'zita-reverb', [p['delay'], p['xover'], p['rtlow'], p['rtmid'], p['damping'], None, None, None, None, p['mix']]))
reg('rev_tap', 'reverb', 'test', lambda r: dict(decay=LU(r, 400, 4000), wet=U(r, -14, 4), type=float(r.integers(0, 43))),
    lambda x, p: run_ladspa(x, 'tap_reverb.so', 'tap_reverb', [p['decay'], 0, p['wet'], 1, 1, 1, 1, p['type']]))
reg('rev_caps_plate', 'reverb', 'dev', lambda r: dict(bandwidth=U(r, 0.3, 0.95), tail=U(r, 0.2, 0.9), damping=U(r, 0.1, 0.8), blend=U(r, 0.2, 0.85)),
    lambda x, p: run_ladspa(x, 'caps.so', 'PlateX2', [p['bandwidth'], p['tail'], p['damping'], p['blend']]))
def _gverb(x, p):
    w = run_ladspa(x, 'gverb_1216.so', 'gverb', [p['room'], p['rt'], p['damp'], p['bw'], -70, p['early'], p['tail']]); return x + w
reg('rev_gverb', 'reverb', 'dev', lambda r: dict(room=LU(r, 8, 120), rt=LU(r, 0.4, 5), damp=U(r, 0.2, 0.9), bw=U(r, 0.4, 0.95), early=U(r, -24, 0), tail=U(r, -22, 2)), _gverb)
reg('rev_swh_plate', 'reverb', 'dev', lambda r: dict(time=LU(r, 0.5, 3), damp=U(r, 0.05, 0.7), mix=U(r, 0.2, 0.9)),
    lambda x, p: run_ladspa(x, 'plate_1423.so', 'plate', [p['time'], p['damp'], p['mix']]))
reg('rev_sox', 'reverb', 'dev', lambda r: dict(reverberance=U(r, 20, 95), hf=U(r, 10, 90), room=U(r, 30, 100), stereo=U(r, 50, 100), predelay=U(r, 0, 40), wet=U(r, -8, 8)),
    lambda x, p: run_sox(x, ['gain', '-3', 'reverb', p['reverberance'], p['hf'], p['room'], p['stereo'], p['predelay'], p['wet']]))
def _pb(board):
    return lambda x, p: pb.Pedalboard(board(p))(x.T.astype(np.float32), SR, reset=True).T.astype(np.float64)
reg('rev_pedalboard', 'reverb', 'dev', lambda r: dict(room_size=U(r, 0.2, 0.95), damping=U(r, 0.1, 0.9), wet_level=LU(r, 0.1, 0.9), dry_level=0.7, width=U(r, 0.5, 1)),
    _pb(lambda p: [pb.Reverb(room_size=p['room_size'], damping=p['damping'], wet_level=p['wet_level'], dry_level=p['dry_level'], width=p['width'])]))

# simulated rooms (image source method): a pair of microphones in a shoebox room
_room_cache = {}
def _room_ir(p):
    import pyroomacoustics as pra
    key = tuple(sorted(p.items()))
    if key in _room_cache: return _room_cache[key]
    dims = [p['lx'], p['ly'], p['lz']]; e_abs, order = pra.inverse_sabine(p['rt60'], dims)
    room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(e_abs), max_order=min(order, 60), air_absorption=True)
    mx, my = p['mx'] * dims[0], p['my'] * dims[1]; r, th = p['dist'], p['th']
    for k in range(64):                                    # the source sits dist metres from the microphones; turn (then come closer) until it is inside the room
        sx, sy = mx + r * np.cos(th), my + r * np.sin(th)
        if 0.3 < sx < dims[0] - 0.3 and 0.3 < sy < dims[1] - 0.3: break
        th += 0.4
        if k % 16 == 15: r *= 0.7
    src = [sx, sy, 1.2 + 0.3 * p['sz']]
    room.add_source(src); room.add_microphone_array(np.array([[mx - 0.09, my, 1.5], [mx + 0.09, my, 1.5]]).T)
    room.compute_rir(); a, b = room.rir[0][0], room.rir[1][0]; n = max(len(a), len(b)); h = np.zeros((n, 2)); h[:len(a), 0] = a; h[:len(b), 1] = b
    k = int(np.argmax(np.abs(h).sum(1))); h = h[max(0, k - 24):]; h /= np.abs(h).max() + 1e-12       # the direct sound at the start, level 1
    _room_cache[key] = h; return h
def _room(x, p):
    h = _room_ir(p); y = np.stack([ss.fftconvolve(x[:, c], h[:, c])[:len(x)] for c in range(2)], 1); return y
def _room_s(lo, hi, rlo, rhi, dhi):
    return lambda r: dict(lx=U(r, lo, hi), ly=U(r, lo, hi) * U(r, 0.6, 1.0), lz=U(r, 2.4, 2.4 + (hi - lo) * 0.3), rt60=LU(r, rlo, rhi), mx=U(r, 0.3, 0.7), my=U(r, 0.3, 0.7), dist=LU(r, 0.25, dhi), th=U(r, 0, 6.283), sz=U(r, 0, 1))
reg('rev_room_small', 'reverb', 'dev', _room_s(2.5, 5, 0.2, 0.6, 2.0), _room)
reg('rev_room_medium', 'reverb', 'dev', _room_s(5, 10, 0.4, 1.2, 4.0), _room)
reg('rev_room_large', 'reverb', 'test', _room_s(10, 25, 0.8, 2.5, 8.0), _room)

# ------------------------------------------------------------------ tone (fixed filtering: EQ, resonances, comb, speaker boxes)
def _eq_bands(r, n, gmax, qlo, qhi):
    return [dict(f=LU(r, 60, 12000), g=float(r.choice([-1, 1])) * U(r, 3, gmax), q=LU(r, qlo, qhi)) for _ in range(n)]
def _pb_eq(x, p):
    board = [pb.PeakFilter(cutoff_frequency_hz=b['f'], gain_db=b['g'], q=b['q']) for b in p['bands']]
    if p.get('ls'): board.append(pb.LowShelfFilter(cutoff_frequency_hz=p['ls']['f'], gain_db=p['ls']['g'], q=0.707))
    if p.get('hs'): board.append(pb.HighShelfFilter(cutoff_frequency_hz=p['hs']['f'], gain_db=p['hs']['g'], q=0.707))
    return pb.Pedalboard(board)(x.T.astype(np.float32), SR, reset=True).T.astype(np.float64)
reg('tone_pb_eq', 'tone', 'dev', lambda r: dict(bands=_eq_bands(r, int(r.integers(1, 5)), 12, 0.6, 6), ls=dict(f=LU(r, 80, 300), g=U(r, -9, 9)) if r.random() < 0.5 else None, hs=dict(f=LU(r, 3000, 10000), g=U(r, -9, 9)) if r.random() < 0.5 else None), _pb_eq)
reg('tone_pb_resonances', 'tone', 'dev', lambda r: dict(bands=[dict(f=LU(r, 120, 9000), g=U(r, 6, 15), q=LU(r, 8, 40)) for _ in range(int(r.integers(2, 7)))]), _pb_eq)
def _sox_eq(x, p):
    fx = ['gain', '-12']
    for b in p['bands']: fx += ['equalizer', b['f'], '%gq' % b['q'], b['g']]
    if p.get('bass') is not None: fx += ['bass', p['bass']]
    if p.get('treble') is not None: fx += ['treble', p['treble']]
    return run_sox(x, fx) * 10 ** (12 / 20)
reg('tone_sox_eq', 'tone', 'test', lambda r: dict(bands=_eq_bands(r, int(r.integers(1, 4)), 12, 0.6, 5), bass=U(r, -10, 10) if r.random() < 0.5 else None, treble=U(r, -10, 10) if r.random() < 0.5 else None), _sox_eq)
def _calf_eq(x, p):
    d = dict(level_in=0.25, level_out=4.0)
    for i, b in enumerate(p['bands'][:3]): d.update({'p%d_active' % (i + 1): 1, 'p%d_level' % (i + 1): 10 ** (b['g'] / 20), 'p%d_freq' % (i + 1): b['f'], 'p%d_q' % (i + 1): b['q']})
    if p.get('ls'): d.update(ls_active=1, ls_level=10 ** (p['ls']['g'] / 20), ls_freq=p['ls']['f'])
    if p.get('hs'): d.update(hs_active=1, hs_level=10 ** (p['hs']['g'] / 20), hs_freq=p['hs']['f'])
    return run_lv2(x, 'http://calf.sourceforge.net/plugins/Equalizer5Band', d)
reg('tone_calf_eq', 'tone', 'dev', lambda r: dict(bands=_eq_bands(r, int(r.integers(1, 4)), 12, 0.6, 6), ls=dict(f=LU(r, 80, 300), g=U(r, -9, 9)) if r.random() < 0.5 else None, hs=dict(f=LU(r, 3000, 10000), g=U(r, -9, 9)) if r.random() < 0.5 else None), _calf_eq)
def _fil4(x, p):
    d = dict(gain=-6.0); rng = [(20, 2000), (40, 4000), (100, 10000), (200, 20000)]
    for i in range(4):
        b = p['bands'][i] if i < len(p['bands']) else None
        if b is None: d['sec%d' % (i + 1)] = 0; continue
        d.update({'sec%d' % (i + 1): 1, 'freq%d' % (i + 1): float(np.clip(b['f'], *rng[i])), 'q%d' % (i + 1): float(np.clip(b['q'], 0.0625, 4)), 'gain%d' % (i + 1): b['g']})
    d.update(LSsec=0, HSsec=0)
    return run_lv2(x, 'http://gareus.org/oss/lv2/fil4#stereo', d) * 2.0
reg('tone_fil4', 'tone', 'test', lambda r: dict(bands=sorted(_eq_bands(r, int(r.integers(1, 5)), 12, 0.5, 4), key=lambda b: b['f'])), _fil4)
def _comb(x, p):   # one short reflection: a comb filter (a hard surface near the microphone, two microphones out of phase...)
    d = p['ms'] * 1e-3 * SR; n = len(x); X = np.fft.rfft(x, axis=0); w = 2 * np.pi * np.fft.rfftfreq(n) * d
    return np.fft.irfft(X * (1 + p['g'] * np.exp(-1j * w))[:, None], n, axis=0)
reg('tone_comb', 'tone', 'dev', lambda r: dict(ms=LU(r, 0.6, 12), g=float(r.choice([-1, 1])) * U(r, 0.3, 0.9)), _comb)
reg('tone_cabinet', 'tone', 'test', lambda r: dict(c_model=float(r.integers(0, 19)), CBass=U(r, -4, 4), CTreble=U(r, -4, 4)),
    lambda x, p: run_lv2(x * 0.5, 'http://guitarix.sourceforge.net/plugins/gx_cabinet#CABINET', dict(p, CLevel=1.0)) * 2.0)
def _box(x, p):   # a tiny room, early reflections only: boxy colouring with no audible tail
    h = _room_ir(dict(p, rt60=p['rt60']))[: int(0.03 * SR)].copy(); h[int(0.02 * SR):] *= np.linspace(1, 0, len(h) - int(0.02 * SR))[:, None]
    return np.stack([ss.fftconvolve(x[:, c], h[:, c])[:len(x)] for c in range(2)], 1)
reg('tone_box', 'tone', 'dev', lambda r: dict(lx=U(r, 1.2, 2.5), ly=U(r, 1.0, 2.2), lz=U(r, 1.8, 2.4), rt60=U(r, 0.12, 0.25), mx=U(r, 0.35, 0.65), my=U(r, 0.35, 0.65), dist=U(r, 0.3, 0.9), th=U(r, 0, 6.283), sz=0.0), _box)

PRE = 1.5     # seconds of silence run through every effect before the sound: several plugins fade their settings in during the first blocks
def apply(name, x, params, tail=0.0, pre=PRE):
    """Runs one zoo effect; tail (s) of silence is added so reverbs and echoes can ring out, and pre (s) of silence goes in front
    (and is cut off again) so the effect has settled on its settings when the sound starts."""
    k = int(pre * SR); n = len(x) + int(tail * SR); xin = np.vstack([np.zeros((k, 2)), x, np.zeros((n - len(x), 2))]); y = ZOO[name][3](xin, params)
    assert y.shape == xin.shape, (name, y.shape, xin.shape); return y[k:]
