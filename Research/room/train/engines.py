# Reverbs for TRAINING: the scoreboard's dev plugins, five more plugins, simulated rooms from the bank, and statistical
# impulse responses (shaped noise with early reflections). The scoreboard's test reverbs are not here.
import sys, glob, json
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, scipy.signal as ss
from zoo import ZOO, run_lv2, run_ladspa, PRE, SR
U = lambda r, a, b: float(r.uniform(a, b)); LU = lambda r, a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
ENG = {}
for name, (fam, split, sampler, fn) in ZOO.items():
    if fam == 'reverb' and split == 'dev' and not name.startswith('rev_room'): ENG[name] = (sampler, fn)
ENG['tr_freeverb3'] = (lambda r: dict(room=U(r, 0.2, 0.98), damp=U(r, 0.0, 0.9), wet=LU(r, 0.05, 0.6), dry=0.7, width=U(r, 0.0, 1.0)),
    lambda x, p: run_ladspa(x, 'cmt.so', 'freeverb3', [0, p['room'], p['damp'], p['wet'], p['dry'], p['width']]))
ENG['tr_caps_plate1'] = (lambda r: dict(bandwidth=U(r, 0.3, 0.95), tail=U(r, 0.1, 0.92), damping=U(r, 0.05, 0.8), blend=U(r, 0.15, 0.85)),
    lambda x, p: run_ladspa(x, 'caps.so', 'Plate', [p['bandwidth'], p['tail'], p['damping'], p['blend']]))
ENG['tr_gx_reverb'] = (lambda r: dict(dry_wet=U(r, 12, 75), roomsize=U(r, 0.2, 0.98), damp=U(r, 0.05, 0.8), mode=0.0),
    lambda x, p: run_lv2(x, 'http://guitarix.sourceforge.net/plugins/gx_reverb_stereo#_reverb_stereo', p))
ENG['tr_gx_room'] = (lambda r: dict(EFFECT=1.0, PREDELAYMS=U(r, 1, 60), RT=U(r, 0.1, 0.95), ROOMSIZE=U(r, 0.3, 3.0), DRYWET=U(r, 0.15, 0.75)),
    lambda x, p: run_lv2(x, 'http://guitarix.sourceforge.net/plugins/gx_room_simulator_#_room_simulator_', p))
def _invada(x, p):
    w = run_ladspa(x, 'inv_erreverb.so', 'invada_sum_reverbER_module_0_1', [p['len'], p['wid'], p['hei'], p['slr'], p['sfb'], p['llr'], p['lfb'], p['hpf'], p['warm'], p['diff']]); return x + p['g'] * w
ENG['tr_invada_er'] = (lambda r: dict(len=LU(r, 4, 60), wid=LU(r, 4, 50), hei=U(r, 3, 15), slr=U(r, 0.1, 0.9), sfb=U(r, 0.56, 0.99), llr=U(r, 0.1, 0.9), lfb=U(r, 0.01, 0.44), hpf=LU(r, 60, 400), warm=U(r, 0.1, 0.9), diff=U(r, 0.1, 0.95), g=LU(r, 0.3, 3.0)), _invada)
PLUGINS = list(ENG.keys())
def run_plugin(name, x, p):
    k = int(PRE * SR); xin = np.vstack([np.zeros((k, 2)), x]); y = ENG[name][1](xin, p); return y[k:]
def plugin_direct(name, p):
    """The direct path of a plugin: its response to a click on each input, the few ms around the click. Returns taps[in] (L,2), offset."""
    n = int(0.3 * SR); at = int(0.1 * SR); h = []
    for i in range(2):
        x = np.zeros((n, 2)); x[at, i] = 1.0; h.append(run_plugin(name, x, p))
    e = (h[0] ** 2 + h[1] ** 2).sum(1); pk = int(np.argmax(e)); a, b = max(0, pk - 48), pk + 120; top = max(np.abs(h[0][a:b]).max(), np.abs(h[1][a:b]).max()); taps = []
    for i in range(2):
        t = h[i][a:b].copy()
        for o in range(2):
            if np.abs(t[:, o]).max() < 0.2 * top: t[:, o] = 0.0                 # a path with no direct sound of its own (only reverb crosses over)
        taps.append(t)
    return taps, a - at
# ---------------------------------------------------------------- simulated rooms from the bank
ROOMS = sorted(glob.glob('/tmp/claude-0/v2/train/rooms/room_*.npz'))
def room_ir(i):
    h = np.load(ROOMS[i % len(ROOMS)])['h'].astype(np.float64); return [h[0], h[1]]
def ir_direct(h):
    """Direct sound of a 2x2 response: for every path the few ms around its own first strong arrival."""
    taps = [np.zeros_like(h[0]), np.zeros_like(h[1])]
    for i in range(2):
        for o in range(2):
            v = np.abs(h[i][:, o]); m = v[:int(0.05 * SR)].max() if len(v) else 0.0
            if m <= 0 or m < 0.05 * max(np.abs(h[0]).max(), np.abs(h[1]).max()): continue
            first = int(np.argmax(v > 0.5 * m)); pk = first + int(np.argmax(v[first:first + 48])); a, b = max(0, pk - 48), pk + 120; taps[i][a:b, o] = h[i][a:b, o]
    return taps
def conv22(x, h, n):
    return np.stack([ss.fftconvolve(x[:, 0], h[0][:, o])[:n] + ss.fftconvolve(x[:, 1], h[1][:, o])[:n] for o in range(2)], 1)
# ---------------------------------------------------------------- statistical responses
def stat_ir(rng):
    t60 = LU(rng, 0.25, 6.0); a = U(rng, 0.0, 0.5); lowm = U(rng, 0.7, 1.8); fd = LU(rng, 2000, 14000); n = int(min(7.0, 1.2 * t60 * max(1.0, lowm)) * SR) + 4800; nf, hop = 1024, 256
    f = np.maximum(np.fft.rfftfreq(nf, 1 / SR), 30.0); T = t60 * (f / 1000.0) ** (-a) * (1 + (lowm - 1) / (1 + (f / 250.0) ** 2)) / (1 + (f / fd) ** 2); T = np.clip(T, 0.05, 12.0)
    tilt = U(rng, -3.0, 1.0); col = (f / 1000.0) ** (tilt / 6.02) / np.sqrt(1 + (f / LU(rng, 5000, 20000)) ** 4)        # colour of the tail, dB per octave and a top roll-off
    corr = float(rng.choice([0.0, 0.0, U(rng, 0, 1), 1.0])); stereo_in = rng.random() < 0.4; w = ss.get_window('hann', nf); frames = 1 + (n - nf) // hop
    def tail():
        com = rng.standard_normal(n); out = np.zeros((n, 2))
        for c in range(2):
            v = np.sqrt(corr) * com + np.sqrt(1 - corr) * rng.standard_normal(n)
            if rng.random() < 0.6:                                             # sparse at first, dense later: single reflections early on
                tm = U(rng, 0.01, 0.08) * SR; dens = np.clip((np.arange(n) / tm) ** 2, 0.02, 1.0); v = v * (rng.random(n) < dens) / np.sqrt(dens)
            idx = np.arange(nf)[None, :] + hop * np.arange(frames)[:, None]; V = np.fft.rfft(v[idx] * w, axis=1)
            V *= np.exp(-6.908 * (np.arange(frames)[:, None] * hop / SR) / T[None, :]) * col[None, :]
            fr = np.fft.irfft(V, nf, axis=1) * w; y = np.zeros(n)
            for t in range(frames): y[t * hop:t * hop + nf] += fr[t]
            out[:, c] = y
        tb = U(rng, 0.0, 0.05) if rng.random() < 0.5 else 0.0
        if tb > 0: k = int(tb * SR); out[:k] *= np.sin(0.5 * np.pi * np.arange(k) / k)[:, None] ** 2
        for _ in range(int(rng.integers(0, 13))):                              # a few single early reflections
            k = int(U(rng, 0.003, 0.07) * SR); g = U(rng, 0.15, 0.8) * rng.choice([-1, 1]) * np.exp(-k / SR / 0.05); pn = U(rng, 0, 1); out[k, 0] += g * np.sqrt(pn) * 8; out[k, 1] += g * np.sqrt(1 - pn) * 8
        return out
    h = [tail(), tail()] if stereo_in else None
    if h is None: t = tail(); h = [0.5 * t, 0.5 * t.copy()]                    # both inputs feed one reverb (most plugins do this)
    drr = U(rng, -8.0, 20.0); e = (h[0] ** 2).sum() + (h[1] ** 2).sum() + ((h[0] * h[1]).sum() * 2 if not stereo_in else 0.0)
    g = np.sqrt(2.0 * 10 ** (-drr / 10) / max(e, 1e-12)); pd = int(U(rng, 0.0, 0.06) * SR) if rng.random() < 0.6 else 0
    out = []
    for i in range(2):
        v = np.zeros((n + pd + 1, 2)); v[pd + 1:] = g * h[i]; v[0, i] = 1.0; out.append(v)
    return out, dict(t60=t60, drr=drr, corr=corr, stereo_in=bool(stereo_in), predelay=pd / SR)
