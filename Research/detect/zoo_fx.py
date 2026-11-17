# Outside effects for the detection scoreboard: one or more implementations per effect family the app names, none of them this
# project's own DSP (LV2 and LADSPA plugins through the bench hosts, sox and ffmpeg built-ins, pedalboard).
# Each entry: name -> (family, split, sampler(rng) -> params, apply(x, params) -> y, clip_filter or None).  x, y: (n, 2) at 48 kHz.
import sys, os, subprocess, shutil, tempfile
import paths; sys.path.insert(0, paths.BENCH)
import numpy as np, scipy.signal as ss, soundfile as sf
import zoo_core
paths.make_work(); zoo_core.TMP = paths.WORK + 'tmp/zoo'     # the runners' scratch files stay in this work's folder
from zoo_core import run_lv2, run_ladspa, run_sox, SR, impulse
import pedalboard as pb

ZOO = {}
def reg(name, family, split, sampler, fn, clips=None): ZOO[name] = (family, split, sampler, fn, clips)
U = lambda r, a, b: float(r.uniform(a, b))
LU = lambda r, a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
CH = lambda r, v: (lambda c: c.item() if hasattr(c, 'item') else c)(r.choice(v))
C = 'http://calf.sourceforge.net/plugins/'; M = 'http://drobilla.net/plugins/mda/'; G = 'http://guitarix.sourceforge.net/plugins/'
def mono(x): m = x.mean(1); return np.stack([m, m], 1)
def _pb(board): return lambda x, p: pb.Pedalboard(board(p))(x.T.astype(np.float32), SR, reset=True).T.astype(np.float64)
def run_ffmpeg(x, args):
    d = tempfile.mkdtemp(dir=zoo_core.TMP); a = os.path.join(d, 'in.wav'); b = os.path.join(d, 'out.wav'); sf.write(a, x.astype(np.float32), SR, subtype='FLOAT')
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a] + [str(v) for v in args] + ['-c:a', 'pcm_f32le', '-ar', str(SR), b], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(b): shutil.rmtree(d, ignore_errors=True); raise RuntimeError('ffmpeg failed: ' + r.stderr[-400:])
    return zoo_core._back(d, b, len(x))

# ------------------------------------------------------------------ flanger
reg('flg_calf', 'flanger', 'dev', lambda r: dict(min_delay=LU(r, 0.2, 3), mod_depth=LU(r, 1, 8), mod_rate=LU(r, 0.1, 2), feedback=U(r, -0.8, 0.8), stereo=CH(r, [0.0, 0.0, 90.0, 180.0]), amount=U(r, 0.5, 1.0), dry=1.0),
    lambda x, p: run_lv2(x * 0.5, C + 'Flanger', p) * 2.0)
reg('flg_sox', 'flanger', 'dev', lambda r: dict(delay=U(r, 0, 4), depth=U(r, 1.5, 8), regen=U(r, -70, 70), width=U(r, 50, 100), speed=LU(r, 0.1, 2), shape=CH(r, ['sine', 'triangle']), phase=CH(r, [0, 25, 50])),
    lambda x, p: run_sox(x, ['gain', '-9', 'flanger', p['delay'], p['depth'], p['regen'], p['width'], p['speed'], p['shape'], p['phase'], 'linear']) * 10 ** (9 / 20))
reg('flg_swh', 'flanger', 'test', lambda r: dict(base=LU(r, 0.3, 5), slow=LU(r, 1, 8), rate=LU(r, 0.1, 2), fb=U(r, -0.7, 0.7)),
    lambda x, p: run_ladspa(x * 0.5, 'flanger_1191.so', 'flanger', [p['base'], p['slow'], p['rate'], p['fb']]) * 2.0)
reg('flg_mda_thruzero', 'flanger', 'test', lambda r: dict(rate=U(r, 0.5, 0.92), depth=U(r, 0.2, 0.5), mix=U(r, 0.3, 0.6), feedback=U(r, 0.25, 0.8), depth_mod=1.0),
    lambda x, p: run_lv2(x, M + 'ThruZero', p))

# ------------------------------------------------------------------ chorus
reg('cho_calf', 'chorus', 'dev', lambda r: dict(min_delay=U(r, 3, 10), mod_depth=U(r, 1, 8), mod_rate=LU(r, 0.15, 2.5), stereo=CH(r, [90.0, 180.0]), voices=float(r.integers(2, 6)), amount=U(r, 0.4, 1.0), dry=1.0),
    lambda x, p: run_lv2(x * 0.5, C + 'MultiChorus', p) * 2.0)
def _sox_chorus(x, p):
    fx = ['gain', '-6', 'chorus', p['gin'], 0.9]
    for v in p['voices']: fx += [v['delay'], v['decay'], v['speed'], v['depth'], '-' + v['shape']]
    return run_sox(x, fx) * 2.0
reg('cho_sox', 'chorus', 'dev', lambda r: dict(gin=U(r, 0.6, 0.8), voices=[dict(delay=U(r, 20, 60), decay=U(r, 0.25, 0.5), speed=LU(r, 0.2, 2.5), depth=U(r, 1, 6), shape=CH(r, ['s', 't'])) for _ in range(int(r.integers(1, 4)))]), _sox_chorus)
reg('cho_pedalboard', 'chorus', 'test', lambda r: dict(rate_hz=LU(r, 0.2, 2.5), depth=U(r, 0.15, 0.6), centre_delay_ms=U(r, 7, 25), feedback=U(r, 0, 0.3), mix=U(r, 0.3, 0.6)),
    _pb(lambda p: [pb.Chorus(**p)]))
reg('cho_tap', 'chorus', 'test', lambda r: dict(freq=LU(r, 0.3, 2.5), phase=CH(r, [90.0, 180.0]), depth=U(r, 30, 90), delay=U(r, 12, 35), contour=100.0, dry=0.0, wet=U(r, -6, 0)),
    lambda x, p: run_ladspa(x * 0.5, 'tap_chorusflanger.so', 'tap_chorusflanger', [p['freq'], p['phase'], p['depth'], p['delay'], p['contour'], p['dry'], p['wet']]) * 2.0)

# ------------------------------------------------------------------ phaser
reg('pha_calf', 'phaser', 'dev', lambda r: dict(base_freq=LU(r, 300, 1500), mod_depth=U(r, 1500, 5000), mod_rate=LU(r, 0.1, 2), feedback=U(r, -0.3, 0.75), stages=float(CH(r, [2, 4, 6, 8, 12])), stereo=CH(r, [0.0, 0.0, 180.0]), amount=U(r, 0.6, 1.0), dry=1.0),
    lambda x, p: run_lv2(x * 0.5, C + 'Phaser', p) * 2.0)
reg('pha_swh', 'phaser', 'dev', lambda r: dict(rate=LU(r, 0.1, 2), depth=U(r, 0.3, 0.9), fb=U(r, -0.3, 0.7), spread=U(r, 0.5, 2)),
    lambda x, p: run_ladspa(x, 'phasers_1217.so', 'lfoPhaser', [p['rate'], p['depth'], p['fb'], p['spread']]))
reg('pha_pedalboard', 'phaser', 'test', lambda r: dict(rate_hz=LU(r, 0.1, 2), depth=U(r, 0.4, 0.95), centre_frequency_hz=LU(r, 400, 2500), feedback=U(r, 0, 0.7), mix=U(r, 0.4, 0.6)),
    _pb(lambda p: [pb.Phaser(**p)]))
reg('pha_caps', 'phaser', 'test', lambda r: dict(rate=U(r, 0.1, 0.7), depth=U(r, 0.5, 1), spread=U(r, 0.3, 1), resonance=U(r, 0, 0.7)),
    lambda x, p: run_ladspa(x, 'caps.so', 'PhaserII', [p['rate'], 0, p['depth'], p['spread'], p['resonance']]))

# ------------------------------------------------------------------ haas (one side late: the mono sum goes to both sides, one of them delayed)
def _sox_haas(x, p):
    d = '%.6f' % (p['ms'] * 1e-3); a = '1v0.5,2v0.5'; b = '1v%.4f,2v%.4f' % (0.5 * p['late_gain'], 0.5 * p['late_gain'])
    return run_sox(x, ['remix', a, b, 'delay', '0', d] if p['late'] == 'R' else ['remix', b, a, 'delay', d, '0'])
reg('haas_sox', 'haas', 'dev', lambda r: dict(ms=LU(r, 3, 35), late=CH(r, ['L', 'R']), late_gain=U(r, 0.7, 1.0)), _sox_haas)
# this plugin's "delay" setting gives half its value as the time between the sides (measured with a click): ms is what comes out
reg('haas_calf_stereotools', 'haas', 'dev', lambda r: (lambda ms, late: dict(ms=ms, late=late))(LU(r, 3, 10), CH(r, ['L', 'R'])),
    lambda x, p: run_lv2(x, C + 'StereoTools', dict(mode=5.0, delay=(2.0 if p['late'] == 'R' else -2.0) * p['ms'])))
reg('haas_ffmpeg', 'haas', 'test', lambda r: dict(ms=LU(r, 3, 35), late=CH(r, ['L', 'R']), late_gain=U(r, 0.7, 1.0)),
    lambda x, p: run_ffmpeg(x, ['-af', 'pan=stereo|c0=%.4f*c0+%.4f*c1|c1=%.4f*c0+%.4f*c1,adelay=%s' % ((0.5 * p['late_gain'],) * 2 + (0.5,) * 2 + ('%.4f|0' % p['ms'],) if p['late'] == 'L' else (0.5,) * 2 + (0.5 * p['late_gain'],) * 2 + ('0|%.4f' % p['ms'],))]))
reg('haas_mda', 'haas', 'test', lambda r: dict(width=U(r, 0.03, 0.2), delay=U(r, 0.25, 0.75), balance=0.5, mod=0.0, rate=0.5),
    lambda x, p: run_lv2(mono(x), M + 'Stereo', p))

# ------------------------------------------------------------------ width (a mono or narrow sound made wide; or the side of a stereo sound raised)
stereo_clip = lambda x: np.sqrt(((x[:, 0] - x[:, 1]) ** 2).mean()) > 0.2 * np.sqrt(((x[:, 0] + x[:, 1]) ** 2).mean())
reg('wid_caps', 'width', 'dev', lambda r: dict(pan=U(r, -0.2, 0.2), width=U(r, 0.4, 1.0)),
    lambda x, p: run_ladspa(x, 'caps.so', 'Wider', [p['pan'], p['width']]) * 2.0)
reg('wid_calf_haasenh', 'width', 'dev', lambda r: dict(m_source=2.0, s_gain=LU(r, 0.3, 1.0), s_delay1=U(r, 1, 9), s_delay2=U(r, 1, 9), s_balance1=-1.0, s_balance2=1.0, s_phase1=0.0, s_phase2=1.0),
    lambda x, p: run_lv2(x, C + 'HaasEnhancer', p))
reg('wid_mda_comb', 'width', 'test', lambda r: dict(width=U(r, 0.7, 1.0), delay=U(r, 0.2, 0.7), balance=0.5, mod=0.0, rate=0.5),
    lambda x, p: run_lv2(mono(x), M + 'Stereo', p))
reg('wid_calf_base', 'width', 'test', lambda r: dict(mode=0.0, stereo_base=U(r, 0.4, 1.0)),
    lambda x, p: run_lv2(x, C + 'StereoTools', p), stereo_clip)

# ------------------------------------------------------------------ delay
def _calf_delay(x, p):
    t = p['beats']; bpm = float(np.clip(15000.0 * t / p['ms'], 30, 300))        # delay = 60/bpm * t/4 s
    return run_lv2(x * 0.5, C + 'VintageDelay', dict(timing=0, bpm=bpm, subdiv=4, time_l=t, time_r=t, feedback=p['feedback'], amount=p['amount'], dry=1.0, mix_mode=p['mix_mode'], medium=p['medium'], width=1.0)) * 2.0
reg('dly_calf', 'delay', 'dev', lambda r: dict(ms=LU(r, 90, 650), beats=float(CH(r, [2, 3, 4])), feedback=U(r, 0.1, 0.65), amount=U(r, 0.25, 0.7), mix_mode=float(CH(r, [0, 0, 1])), medium=float(CH(r, [0, 1, 2]))), _calf_delay)
def _sox_echo(x, p):
    fx = ['gain', '-6', 'echo', 0.9, 0.9]
    for k in range(1, p['taps'] + 1): fx += [p['ms'] * k, p['decay'] ** k]
    return run_sox(x, fx) * 2.0 / 0.81
reg('dly_sox', 'delay', 'dev', lambda r: dict(ms=LU(r, 80, 600), decay=U(r, 0.25, 0.65), taps=int(r.integers(1, 5))), _sox_echo)
reg('dly_pedalboard', 'delay', 'test', lambda r: dict(delay_seconds=LU(r, 0.08, 0.7), feedback=U(r, 0.1, 0.65), mix=U(r, 0.2, 0.5)), _pb(lambda p: [pb.Delay(**p)]))
reg('dly_tap', 'delay', 'test', lambda r: (lambda ms, cross: dict(l_ms=ms, r_ms=ms * CH(r, [1.0, 1.0, 1.5, 0.75]), fb=U(r, 10, 60), level=U(r, -10, -3), cross=cross))(LU(r, 90, 600), float(CH(r, [0, 0, 1]))),
    lambda x, p: run_ladspa(x, 'tap_echo.so', 'tap_stereo_echo', [p['l_ms'], p['fb'], p['r_ms'], p['fb'], p['level'], p['level'], 0, p['cross'], 0, 0]))

# ------------------------------------------------------------------ distortion
reg('dist_pedalboard', 'distortion', 'dev', lambda r: dict(drive_db=U(r, 14, 34)), lambda x, p: _pb(lambda q: [pb.Distortion(**q)])(x, p) * 0.5)
reg('dist_sox', 'distortion', 'dev', lambda r: dict(gain=U(r, 12, 40), colour=U(r, 0, 60)), lambda x, p: run_sox(x, ['overdrive', p['gain'], p['colour']]))
def _caps_sat(x, p):       # 8 samples late (its oversampling), and each curve comes out at its own level: brought back in time and to the dry sound's peak
    y = run_ladspa(x, 'caps.so', 'Saturate', [p['mode'], p['gain'], p['bias']]); y = np.vstack([y[8:], np.zeros((8, 2))]); return y * (np.abs(x).max() / (np.abs(y).max() + 1e-12))
reg('dist_caps', 'distortion', 'test', lambda r: dict(mode=float(CH(r, [1, 2, 3, 5, 9, 10])), gain=U(r, 10, 32), bias=U(r, 0, 0.3)), _caps_sat)
reg('dist_mda', 'distortion', 'test', lambda r: dict(drive=U(r, 0.45, 1.0), muffle=U(r, 0, 0.5), output=0.5), lambda x, p: run_lv2(x, M + 'Overdrive', p))

# ------------------------------------------------------------------ bitcrush
reg('bit_calf', 'bitcrush', 'dev', lambda r: dict(bits=float(r.integers(3, 10)), samples=float(r.integers(1, 10)), morph=U(r, 0.85, 1.0), mode=0.0, anti_aliasing=U(r, 0, 0.3), dc=1.0, lfo=0.0),
    lambda x, p: run_lv2(x, C + 'Crusher', p))
def _pb_crush(x, p):
    board = [pb.Bitcrush(bit_depth=p['bits'])] + ([pb.Resample(target_sample_rate=SR / p['down'], quality=pb.Resample.Quality.ZeroOrderHold)] if p['down'] > 1 else [])
    return pb.Pedalboard(board)((x * 2.0).T.astype(np.float32), SR, reset=True).T.astype(np.float64) * 0.5
reg('bit_pedalboard', 'bitcrush', 'dev', lambda r: dict(bits=float(r.integers(3, 10)), down=int(r.integers(1, 10))), _pb_crush)
reg('bit_swh', 'bitcrush', 'test', lambda r: dict(bits=float(r.integers(3, 10)), down=int(r.integers(1, 10))),
    lambda x, p: run_ladspa(x * 2.0, 'decimator_1202.so', 'decimator', [p['bits'], SR / p['down']]) * 0.5)
reg('bit_mda', 'bitcrush', 'test', lambda r: dict(headroom=0.8, quant=U(r, 0.08, 0.52), rate=0.5 + float(CH(r, [-1, 1])) * U(r, 0.0, 0.12), post_filt=U(r, 0.9, 1.0), non_lin=0.5, output=0.5),
    lambda x, p: -run_lv2(x, M + 'Degrade', p))        # this plugin turns the sound upside down (and makes it mono): turned back

# ------------------------------------------------------------------ filter sweep (a resonant filter whose cutoff an LFO moves)
def _pb_sweep(x, p, blk=256):
    n = len(x); y = np.zeros_like(x); t = np.arange(0, n, blk) / SR; ph = p['rate'] * t + p['phase']
    lfo = np.sin(2 * np.pi * ph) if p['shape'] == 'sine' else 2 * np.abs(2 * (ph % 1) - 1) - 1
    fc = p['fmin'] * (p['fmax'] / p['fmin']) ** (0.5 * (1 + lfo))
    f = pb.LadderFilter(mode=getattr(pb.LadderFilter.Mode, p['mode']), cutoff_hz=float(fc[0]), resonance=p['resonance'], drive=1.0); xs = x.T.astype(np.float32)
    for i, s0 in enumerate(range(0, n, blk)):       # the plugin's cutoff is automated block by block, as a host does it
        f.cutoff_hz = float(fc[i]); y[s0:s0 + blk] = f.process(xs[:, s0:s0 + blk], SR, buffer_size=blk, reset=False).T
    return y
reg('swp_pb_ladder', 'filtersweep', 'dev', lambda r: dict(rate=LU(r, 0.15, 2), phase=U(r, 0, 1), shape=CH(r, ['sine', 'sine', 'tri']), fmin=LU(r, 150, 800), fmax=LU(r, 1500, 9000), resonance=U(r, 0.2, 0.8), mode=CH(r, ['LPF12', 'LPF24'])), _pb_sweep)
reg('swp_pb_bandpass', 'filtersweep', 'dev', lambda r: dict(rate=LU(r, 0.15, 2), phase=U(r, 0, 1), shape=CH(r, ['sine', 'tri']), fmin=LU(r, 200, 600), fmax=LU(r, 1500, 5000), resonance=U(r, 0.3, 0.8), mode=CH(r, ['BPF12', 'BPF24'])), _pb_sweep)
def _mda_swp(r):            # this plugin's cutoff is 1.5 f^2 - 0.15 (of a state-variable filter), the LFO moves it by 2 (d - 0.5)^2 either way, the LFO runs at 10^(3 v - 1.5) Hz
    f = U(r, 0.47, 0.68); mid = 1.5 * f * f - 0.15; swing = U(r, 0.55, 0.95) * (mid - 0.012)
    return dict(freq=f, res=U(r, 0.3, 0.8), output=0.5, env_vcf=0.5, attack=0.0, release=0.5, lfo_vcf=0.5 + float(np.sqrt(swing / 2)), lfo_rate=U(r, 0.23, 0.6), trigger=0.0, max_freq=1.0)
reg('swp_mda', 'filtersweep', 'test', _mda_swp,
    lambda x, p: run_lv2(x, M + 'RezFilter', p))
reg('swp_caps', 'filtersweep', 'test', lambda r: dict(mode=float(CH(r, [0, 1])), filter=float(CH(r, [0, 1])), f=LU(r, 200, 1200), Q=U(r, 0.2, 0.7), depth=U(r, 0.6, 1.0), lfoenv=0.0, rate=U(r, 0.2, 0.7), shape=U(r, 0.5, 1.0)),
    lambda x, p: run_ladspa(x, 'caps.so', 'AutoFilter', [p['mode'], p['filter'], p['f'], p['Q'], p['depth'], p['lfoenv'], p['rate'], p['shape']]))

# ------------------------------------------------------------------ tremolo
reg('trm_sox', 'tremolo', 'dev', lambda r: dict(speed=LU(r, 1.5, 12), depth=U(r, 30, 95)), lambda x, p: run_sox(x, ['tremolo', p['speed'], p['depth']]))
reg('trm_calf', 'tremolo', 'dev', lambda r: dict(mode=float(CH(r, [0, 0, 1])), amount=U(r, 0.3, 1.0), offset_l=0.0, offset_r=0.0, timing=2.0, hz=LU(r, 1.5, 12)),
    lambda x, p: run_lv2(x, C + 'Pulsator', p))
reg('trm_tap', 'tremolo', 'test', lambda r: dict(freq=LU(r, 1.5, 12), depth=U(r, 30, 100)), lambda x, p: run_ladspa(x, 'tap_tremolo.so', 'tap_tremolo', [p['freq'], p['depth'], 0]))
reg('trm_gx', 'tremolo', 'test', lambda r: dict(dry_wet=100.0, mode=float(CH(r, [0, 1])), depth=U(r, 0.3, 1.0), freq=LU(r, 1.5, 12)),
    lambda x, p: run_lv2(x, G + 'gx_tremolo#_tremolo', p))

# ------------------------------------------------------------------ auto pan
reg('pan_tap', 'autopan', 'dev', lambda r: dict(freq=LU(r, 0.15, 3), depth=U(r, 50, 100)), lambda x, p: run_ladspa(x, 'tap_autopan.so', 'tap_autopan', [p['freq'], p['depth'], 0]))
reg('pan_calf', 'autopan', 'dev', lambda r: dict(mode=float(CH(r, [0, 0, 1])), amount=U(r, 0.5, 1.0), offset_l=0.0, offset_r=0.5, timing=2.0, hz=LU(r, 0.15, 3)),
    lambda x, p: run_lv2(x, C + 'Pulsator', p))
def _sox_pan(x, p):     # each side multiplied by a slow sine, the two a half turn apart; with less than full depth the plain sound is mixed back in
    w = run_sox(x, ['synth', 'sine', 'amod', p['rate'], 0, p['phase'], 'sine', 'amod', p['rate'], 0, (p['phase'] + 50) % 100]); return (1 - p['depth']) * x + p['depth'] * w * 1.0
reg('pan_sox', 'autopan', 'test', lambda r: dict(rate=LU(r, 0.15, 3), phase=U(r, 0, 100), depth=U(r, 0.5, 1.0)), _sox_pan)
reg('pan_mda', 'autopan', 'test', lambda r: dict(pan=0.5, auto=float(CH(r, [-1, 1])) * U(r, 0.08, 0.35) + 0.5), lambda x, p: run_lv2(x, M + 'RoundPan', p))

# ------------------------------------------------------------------ saturation
reg('sat_calf', 'saturation', 'dev', lambda r: dict(drive=LU(r, 1.5, 9), blend=U(r, -10, 10), mix=1.0, level_in=2.0, level_out=0.5), lambda x, p: run_lv2(x, C + 'Saturator', p))
reg('sat_tap_tube', 'saturation', 'dev', lambda r: dict(drive=LU(r, 1.5, 10), blend=U(r, -10, 10)), lambda x, p: run_ladspa(x * 2.0, 'tap_tubewarmth.so', 'tap_tubewarmth', [p['drive'], p['blend']]) * 0.5)
reg('sat_swh_valve', 'saturation', 'test', lambda r: dict(level=U(r, 0.3, 1.0), character=U(r, 0, 1)), lambda x, p: run_ladspa(x * 2.0, 'valve_1209.so', 'valve', [p['level'], p['character']]) * 0.5)
reg('sat_calf_tape', 'saturation', 'test', lambda r: dict(level_in=LU(r, 1.0, 6.0), level_out=1.0, mix=1.0, lowpass=LU(r, 6000, 18000), speed=float(CH(r, [0, 1])), noise=0.0, mechanical=0.0, magnetical=1.0, post=0.0),
    lambda x, p: (lambda y: y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-12)))(run_lv2(x, C + 'TapeSimulator', p)))

# ------------------------------------------------------------------ exciter
reg('exc_calf', 'exciter', 'dev', lambda r: dict(amount=LU(r, 0.5, 4), drive=U(r, 3, 10), blend=U(r, -5, 5), freq=LU(r, 2000, 8000)), lambda x, p: run_lv2(x, C + 'Exciter', p))
def _sox_exciter(x, p):     # how an exciter is built, from sox parts: the highs alone are overdriven, cleaned of what fell below, and put under the sound
    h = run_sox(x, ['highpass', '-2', p['freq'], 'gain', p['push'], 'overdrive', p['drive'], 20, 'highpass', '-2', p['freq'], 'highpass', '-2', p['freq']])
    g = np.sqrt((x ** 2).mean()) / (np.sqrt((h ** 2).mean()) + 1e-12); return x + p['amount'] * g * h
reg('exc_sox', 'exciter', 'dev', lambda r: dict(freq=LU(r, 1500, 8000), push=U(r, 6, 18), drive=U(r, 10, 25), amount=U(r, 0.15, 0.7)), _sox_exciter)
reg('exc_caps_spice', 'exciter', 'test', lambda r: dict(hi_f=LU(r, 1500, 5000), hi_gain=U(r, 0.3, 1.0), hi_vol=U(r, -9, 3)),
    lambda x, p: run_ladspa(x, 'caps.so', 'SpiceX2', [100, 0, 0, -60, p['hi_f'], p['hi_gain'], p['hi_vol']]))

# ------------------------------------------------------------------ ring modulator
reg('rng_calf', 'ringmod', 'dev', lambda r: dict(mod_mode=0.0, mod_freq=LU(r, 30, 2000), mod_amount=U(r, 0.4, 1.0), mod_phase=0.0, mod_detune=0.0), lambda x, p: run_lv2(x, C + 'RingModulator', p))
def _sox_ring(x, p): w = run_sox(x, ['synth', 'sine', 'fmod', p['carrier']]); return (1 - p['mix']) * x + p['mix'] * w * np.sqrt(2)
reg('rng_sox', 'ringmod', 'dev', lambda r: dict(carrier=LU(r, 30, 2000), mix=U(r, 0.4, 1.0)), _sox_ring)
reg('rng_swh', 'ringmod', 'test', lambda r: dict(depth=U(r, 1.2, 2.0), freq=LU(r, 30, 1000)), lambda x, p: run_ladspa(x, 'ringmod_1188.so', 'ringmod_1i1o1l', [p['depth'], p['freq'], 1, 0, 0, 0]))
reg('rng_mda', 'ringmod', 'test', lambda r: dict(freq=U(r, 0.0, 0.125), fine=U(r, 0.3, 1), feedback=0.0), lambda x, p: run_lv2(x, M + 'RingMod', p))

# ------------------------------------------------------------------ filter (fixed low-pass, high-pass or band-pass, with or without resonance)
def _fsample(r):
    t = CH(r, ['low', 'low', 'high', 'high', 'band']); return dict(type=t, cutoff=LU(r, 300, 8000) if t == 'low' else LU(r, 150, 3000) if t == 'high' else LU(r, 300, 4000), q=LU(r, 0.6, 7), steep=int(CH(r, [0, 1])))
reg('fil_calf', 'filter', 'dev', _fsample, lambda x, p: run_lv2(x * 0.5, C + 'Filter', dict(freq=p['cutoff'], res=max(0.707, p['q']), mode=dict(low=0, high=3, band=6)[p['type']] + p['steep'], inertia=5.0)) * 2.0)
def _sox_filter(x, p):
    one = [dict(low='lowpass', high='highpass', band='bandpass')[p['type']]] + ([] if p['type'] == 'band' else ['-2']) + [p['cutoff'], '%gq' % p['q']]
    return run_sox(x, ['gain', '-12'] + one + (one[:-1] + ['0.707q'] if p['steep'] else [])) * 4.0
reg('fil_sox', 'filter', 'dev', _fsample, _sox_filter)
reg('fil_pb_ladder', 'filter', 'test', lambda r: (lambda d: dict(d, resonance=U(r, 0.0, 0.85)))(_fsample(r)),
    lambda x, p: _pb(lambda q: [pb.LadderFilter(mode=getattr(pb.LadderFilter.Mode, dict(low='LPF', high='HPF', band='BPF')[q['type']] + ('24' if q['steep'] else '12')), cutoff_hz=q['cutoff'], resonance=q['resonance'], drive=1.0)])(x, p))
reg('fil_swh_ls', 'filter', 'test', lambda r: (lambda d: dict(d, resonance=U(r, 0.0, 0.8)))(_fsample(r)),
    lambda x, p: run_ladspa(x * 0.5, 'ls_filter_1908.so', 'lsFilter', [dict(low=0, band=1, high=2)[p['type']], p['cutoff'], p['resonance']]) * 2.0)

# ------------------------------------------------------------------ stutter (a slice played several times over; no stutter plugin is installed, so it is cut and pasted with sox, and with ffmpeg)
def _stutter_plan(n, p):
    r = np.random.default_rng(p['seed']); L = int(p['slice_ms'] * 1e-3 * SR); pos = 0; plan = []
    while pos + L <= n:
        if r.random() < p['prob']: plan.append((pos, L, p['repeats'])); pos += p['repeats'] * L
        else: pos += L
    return plan
def _sox_stutter(x, p):
    n = len(x); d = tempfile.mkdtemp(dir=zoo_core.TMP); a = os.path.join(d, 'in.wav'); sf.write(a, x.astype(np.float32), SR, subtype='FLOAT'); parts = []; pos = 0
    def cut(s0, L, extra, k):
        o = os.path.join(d, 'p%03d.wav' % k); subprocess.run(['sox', '-D', a, '-e', 'floating-point', '-b', '32', o, 'trim', '%ds' % s0, '%ds' % L] + extra, check=True, capture_output=True); return o
    for s0, L, reps in _stutter_plan(n, p):
        if s0 > pos: parts.append(cut(pos, s0 - pos, [], len(parts)))
        parts.append(cut(s0, L, ['fade', 't', '0.002', '%ds' % L, '0.002', 'repeat', str(reps - 1)], len(parts))); pos = s0 + reps * L
    if pos < n: parts.append(cut(pos, n - pos, [], len(parts)))
    b = os.path.join(d, 'out.wav'); subprocess.run(['sox', '-D'] + parts + ['-e', 'floating-point', '-b', '32', b], check=True, capture_output=True)
    return zoo_core._back(d, b, n)
def _ff_stutter(x, p):
    n = len(x); plan = _stutter_plan(n, p); segs = []; pos = 0; fg = []
    for s0, L, reps in plan:
        if s0 > pos: segs.append('atrim=start_sample=%d:end_sample=%d,asetpts=N/SR/TB' % (pos, s0))
        segs.append('atrim=start_sample=%d:end_sample=%d,asetpts=N/SR/TB,afade=t=in:ns=96,afade=t=out:st=%f:d=0.002,aloop=loop=%d:size=%d:start=0' % (s0, s0 + L, (L - 96) / SR, reps - 1, L)); pos = s0 + reps * L
    if pos < n: segs.append('atrim=start_sample=%d:end_sample=%d,asetpts=N/SR/TB' % (pos, n))
    if len(segs) == 1: return run_ffmpeg(x, ['-af', segs[0]])
    fg = '[0:a]asplit=%d%s;' % (len(segs), ''.join('[i%d]' % i for i in range(len(segs)))) + ';'.join('[i%d]%s[o%d]' % (i, s, i) for i, s in enumerate(segs)) + ';' + ''.join('[o%d]' % i for i in range(len(segs))) + 'concat=n=%d:v=0:a=1[out]' % len(segs)
    return run_ffmpeg(x, ['-filter_complex', fg, '-map', '[out]'])
_stsample = lambda r: dict(slice_ms=LU(r, 30, 250), repeats=int(r.integers(2, 9)), prob=U(r, 0.25, 0.8), seed=int(r.integers(1e6)))
reg('stu_sox', 'stutter', 'dev', _stsample, _sox_stutter)
reg('stu_ffmpeg', 'stutter', 'test', _stsample, _ff_stutter)

# ------------------------------------------------------------------ gated rhythm (the sound switched on and off in a steady pulse)
def _sox_gate(x, p): w = run_sox(x, ['synth', 'square', 'amod', p['rate'], 0, p['phase'], 100 * p['duty']]); return (1 - p['depth']) * x + p['depth'] * w
reg('gat_sox', 'gatedrhythm', 'dev', lambda r: dict(rate=LU(r, 2, 16), duty=U(r, 0.2, 0.8), depth=U(r, 0.6, 1.0), phase=U(r, 0, 100)), _sox_gate)
reg('gat_calf', 'gatedrhythm', 'dev', lambda r: dict(mode=2.0, amount=U(r, 0.6, 1.0), offset_l=0.0, offset_r=0.0, timing=2.0, hz=LU(r, 2, 16)), lambda x, p: run_lv2(x, C + 'Pulsator', p))
def _ff_gate(x, p):         # a noise gate opened by a key signal: a 16-step on/off pattern (the key is only a control signal, as a sequencer would send it)
    n = len(x); t = np.arange(n) / SR; sp = t * p['rate'] + p['phase']; step = np.floor(sp).astype(int) % 16; on = ((p['pattern'] >> step) & 1) * ((sp - np.floor(sp)) < p['duty'])
    key = 0.5 * on * np.sin(2 * np.pi * 1000 * t); d = tempfile.mkdtemp(dir=zoo_core.TMP); a = os.path.join(d, 'in.wav'); k = os.path.join(d, 'key.wav'); b = os.path.join(d, 'out.wav')
    sf.write(a, x.astype(np.float32), SR, subtype='FLOAT'); sf.write(k, np.stack([key, key], 1).astype(np.float32), SR, subtype='FLOAT')
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a, '-i', k, '-filter_complex', '[0:a][1:a]sidechaingate=threshold=0.05:ratio=9000:range=%.4f:attack=1:release=8:detection=peak:knee=1[out]' % max(1 - p['depth'], 0.001),
                        '-map', '[out]', '-c:a', 'pcm_f32le', b], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(b): shutil.rmtree(d, ignore_errors=True); raise RuntimeError('ffmpeg failed: ' + r.stderr[-400:])
    return zoo_core._back(d, b, n)
reg('gat_ffmpeg_keyed', 'gatedrhythm', 'test', lambda r: dict(rate=LU(r, 2, 16), duty=U(r, 0.2, 0.8), depth=U(r, 0.6, 1.0), pattern=(int(r.integers(0, 1 << 16)) | 0x1111) if r.random() < 0.5 else 0xFFFF, phase=U(r, 0, 1)), _ff_gate)

TAIL = { 'delay': 3.0 }      # seconds of silence run through behind the sound, per family (others: none)
PRE = 1.5
def apply(name, x, params, tail=0.0, pre=PRE):
    """Runs one effect; tail (s) of silence behind the sound lets echoes ring out; pre (s) of silence in front (cut off again)
    lets the plugin settle on its settings before the sound starts."""
    k = int(pre * SR); n = len(x) + int(tail * SR); xin = np.vstack([np.zeros((k, 2)), x, np.zeros((n - len(x), 2))]); y = ZOO[name][3](xin, params)
    assert y.shape == xin.shape, (name, y.shape, xin.shape); return y[k:]
