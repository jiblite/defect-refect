# More clean sounds for training the verification step, made from material the scoreboard's corpus does NOT use
# (other General MIDI instruments and other notes, other sample files, other voices and sentences, other synthetic sounds),
# so that none of them shares a source with a test clip. Same format as the corpus: 6 s, 48 kHz stereo, peak 0.5.
#   make(i) -> (name, kind, audio)      deterministic in i, 0 <= i < COUNT
import sys, os, glob, subprocess, hashlib
import paths
import numpy as np, soundfile as sf, scipy.signal as ss, pretty_midi
SR = 48000; N = 6 * SR; paths.make_work(); TMPD = paths.WORK + 'tmp/extra'
# the General MIDI sound font and the two folders of free samples, as their Debian packages install them (as in Research/room/train/mkclips.py)
SF2 = os.environ.get('DR_SF2', '/usr/share/sounds/sf2/FluidR3_GM.sf2'); SP = os.environ.get('DR_SAMPLES_A', '/usr/share/sonic-pi/samples/'); LM = os.environ.get('DR_SAMPLES_B', '/usr/share/lmms/samples/')

def load(path):
    x, sr = sf.read(path, always_2d=True); x = x.astype(np.float64)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    x = x[:, :2]
    if sr != SR: x = ss.resample_poly(x, SR, sr, axis=0)
    return x
def fit(x, n=N, loop=False):
    if len(x) >= n: return x[:n].copy()
    if loop and len(x) > SR // 4: return np.tile(x, (int(np.ceil(n / len(x))), 1))[:n]
    return np.vstack([x, np.zeros((n - len(x), 2))])
def finish(x):
    x = fit(x); x = x - x.mean(0); pk = np.abs(x).max()
    if pk < 1e-4 or (x ** 2).mean() < 1e-9: return None
    x = x * (0.5 / pk); f = np.ones(len(x)); k = int(0.01 * SR); f[:k] = np.linspace(0, 1, k); f[-k:] = np.linspace(1, 0, k); return x * f[:, None]
def render(pm):
    b = os.path.join(TMPD, 'r%d' % os.getpid()); pm.write(b + '.mid')
    subprocess.run(['fluidsynth', '-ni', '-F', b + '.wav', '-r', str(SR), '-g', '0.8', '-o', 'synth.reverb.active=0', '-o', 'synth.chorus.active=0', SF2, b + '.mid'], capture_output=True, check=True)
    x = load(b + '.wav'); os.remove(b + '.wav'); os.remove(b + '.mid'); return x

# General MIDI programs the corpus does not use (and none that carries an effect in its own name: overdriven and distorted guitars,
# tremolo strings, the detuned piano, the sweeping and echoing synth pads, the sound effects)
PROGS = [1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 20, 21, 22, 23, 26, 28, 31, 34, 35, 36, 37, 39, 41, 43, 49, 50, 51, 53, 54, 55, 58, 59, 62, 63, 64, 66, 67, 69, 70,
         72, 74, 75, 76, 77, 78, 79, 82, 83, 85, 87, 88, 91, 92, 93, 94, 104, 105, 106, 107, 109, 110, 111, 112, 113, 117, 118]
LOW = { 34, 35, 36, 37, 39, 43, 58, 67, 70 }; HIGH = { 8, 9, 10, 13, 72, 74, 78, 79, 112 }
SCALES = [[0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10], [0, 2, 4, 7, 9], [0, 3, 5, 6, 7, 10], [0, 2, 3, 5, 7, 9, 10], [0, 1, 4, 5, 7, 8, 11], [0, 2, 4, 6, 8, 10]]
def melody(rng, prog, lo, hi):
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=prog); sc = SCALES[rng.integers(len(SCALES))]; root = int(rng.integers(0, 12))
    pit = [p for p in range(lo, hi) if (p - root) % 12 in sc]; i = int(rng.integers(len(pit))); t = rng.uniform(0.02, 0.3); base = rng.choice([0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6]); leg = rng.uniform(0.3, 1.1)
    two = rng.random() < 0.25
    while t < 5.6:
        d = base * rng.choice([1, 1, 1, 2, 2, 3, 4, 6])
        if rng.random() < 0.18: t += d; continue
        i = int(np.clip(i + rng.integers(-4, 5), 0, len(pit) - 1)); v = int(rng.integers(45, 125))
        inst.notes.append(pretty_midi.Note(velocity=v, pitch=pit[i], start=t, end=min(5.85, t + d * leg)))
        if two and rng.random() < 0.6: inst.notes.append(pretty_midi.Note(velocity=v, pitch=pit[max(0, i - 2)], start=t, end=min(5.85, t + d * leg)))
        t += d
    pm.instruments.append(inst); return pm
def chords(rng, prog):
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=prog); root = int(rng.integers(38, 58)); t = rng.uniform(0.02, 0.2); d = rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]); arp = rng.random() < 0.3
    while t < 5.4:
        deg = rng.choice([0, 2, 3, 5, 7, 8, 9, 10]); third = rng.choice([3, 4]); vel = int(rng.integers(55, 115)); ivs = [0, third, 7] + ([10] if rng.random() < 0.4 else []) + ([14] if rng.random() < 0.2 else []) + [12]
        for j, iv in enumerate(ivs):
            s = t + (j * d / (len(ivs) + 1) if arp else rng.uniform(0, 0.025)); inst.notes.append(pretty_midi.Note(velocity=vel, pitch=root + deg + iv, start=s, end=min(5.85, t + d * rng.uniform(0.5, 1.0))))
        t += d
    pm.instruments.append(inst); return pm
def drums(rng, pm=None):
    pm = pm or pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=int(rng.choice([0, 8, 16, 24, 25, 32, 40, 48])), is_drum=True); bpm = rng.uniform(60, 175); s = 60 / bpm / 4
    sw = rng.uniform(0, 0.3) if rng.random() < 0.3 else 0.0
    kick = rng.random(16) < np.array([.9, .05, .15, .2, .1, .1, .35, .1, .6, .1, .3, .1, .15, .1, .25, .2]); snare = rng.random(16) < np.array([0, 0, .05, .05, .9, 0, .1, .2, 0, .15, 0, .1, .9, 0, .15, .25])
    hat = rng.random(16) < rng.choice([0.3, 0.5, 0.75, 1.0]); k = 0; t = rng.uniform(0.02, 0.2); toms = [45, 47, 50, 49, 51, 56, 54, 37, 39, 60, 61, 62, 63, 64, 69, 70, 75, 76]
    while t < 5.75:
        j = k % 16; tt = t + (sw * s if j % 2 else 0)
        if kick[j]: inst.notes.append(pretty_midi.Note(int(rng.integers(85, 127)), int(rng.choice([35, 36])), tt, tt + 0.1))
        if snare[j]: inst.notes.append(pretty_midi.Note(int(rng.integers(70, 127)), int(rng.choice([38, 40, 39, 37])), tt, tt + 0.1))
        if hat[j] and (j % 2 == 0 or rng.random() < 0.4): inst.notes.append(pretty_midi.Note(int(rng.integers(40, 105)), int(rng.choice([42, 42, 42, 46, 44, 51, 53])), tt, tt + 0.05))
        if rng.random() < 0.06: inst.notes.append(pretty_midi.Note(int(rng.integers(60, 115)), int(rng.choice(toms)), tt, tt + 0.1))
        t += s; k += 1
    pm.instruments.append(inst); return pm
def stabs(rng, prog, lo, hi):
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=prog); t = rng.uniform(0.05, 0.4)
    while t < 5.3:
        p = int(rng.integers(lo, hi))
        for iv in ([0] if rng.random() < 0.5 else [0, 4, 7] if rng.random() < 0.5 else [0, 3, 7, 10]): inst.notes.append(pretty_midi.Note(int(rng.integers(85, 127)), p + iv, t, t + rng.uniform(0.06, 0.35)))
        t += rng.uniform(0.5, 1.8)
    pm.instruments.append(inst); return pm
def rng_of(tag): return np.random.default_rng(int(hashlib.md5(('extra/' + tag).encode()).hexdigest(), 16) % 2 ** 32)
def span(prog): return (28, 55) if prog in LOW else (60, 96) if prog in HIGH else (45, 84)

def k_fluid_mel(j): prog = PROGS[(j * 7) % len(PROGS)]; r = rng_of('mel%d' % j); return render(melody(r, prog, *span(prog)))
def k_fluid_chd(j): prog = PROGS[(j * 11 + 3) % len(PROGS)]; r = rng_of('chd%d' % j); return render(chords(r, prog))
def k_fluid_stab(j): prog = PROGS[(j * 5 + 1) % len(PROGS)]; r = rng_of('stab%d' % j); return render(stabs(r, prog, *span(prog)))
def k_fluid_band(j):
    r = rng_of('band%d' % j); pm = drums(r)
    if j % 3:                                   # two of three with a bass and a second instrument, one of three drums alone
        lowp = sorted(LOW); b = melody(r, int(r.choice(lowp)), 28, 50).instruments[0]; k = (chords if r.random() < 0.6 else lambda rr, p: melody(rr, p, 48, 84))(r, PROGS[int(r.integers(len(PROGS)))]).instruments[0]
        for n in b.notes: n.velocity = min(127, n.velocity + 10)
        pm.instruments += [b, k]
    return render(pm)

def _files():
    sp = lambda pat: sorted(glob.glob(SP + pat)); lm = lambda d: sorted(glob.glob(LM + d + '/*.ogg'))
    loops = [f for f in sp('ambi_*.flac') if 'woosh' not in f and 'swoosh' not in f] + sp('bass_*.flac')[4:] + sp('mehackit_*.flac')[4:] + sp('vinyl_*.flac') + sp('misc_*.flac') + lm('beats')[8:] + lm('bassloops')[5:] + lm('latin')
    notes = (lm('instruments') + lm('stringsnpads'))[10:] + lm('basses'); shots = lm('misc') + sp('vinyl_*.flac') + sp('misc_*.flac')
    waves = sorted(glob.glob(LM + 'waveforms/*.flac')) + sorted(glob.glob(LM + 'shapes/*.wav'))
    return loops, notes, shots, waves
_F = {}
def files():
    if not _F: _F['v'] = _files()
    return _F['v']
def _try(f):
    try: return load(f)
    except Exception: return None
def k_loop(j):
    loops = files()[0]; x = _try(loops[j % len(loops)])
    if x is None: return None
    if j >= len(loops):                            # second round: from another place in the file, or at another speed
        r = rng_of('loop%d' % j)
        if len(x) > N + SR: s = int(r.integers(0, len(x) - N)); x = x[s:]
        else: x = ss.resample(x, int(len(x) * 2 ** (r.choice([-3, -2, 2, 3, 4]) / 12)), axis=0)
    return fit(x, loop=True)
def k_notes(j):
    notes = files()[1]; r = rng_of('notes%d' % j); x = _try(notes[j % len(notes)])
    if x is None: return None
    y = np.zeros((N + 4 * SR, 2)); t = r.uniform(0.02, 0.3); x = x[:min(len(x), 2 * SR)]
    while t < 5.2:
        q = 2 ** (r.integers(-9, 10) / 12); h = ss.resample(x, max(64, int(len(x) / q)), axis=0); L = min(len(h), int(r.uniform(0.2, 1.5) * SR)); w = np.ones(L); w[-min(L, 2400):] = np.linspace(1, 0, min(L, 2400))
        i = int(t * SR); y[i:i + L] += h[:L] * w[:, None] * r.uniform(0.4, 1.0); t += r.uniform(0.2, 1.1)
    return y[:N]
def k_seq(j):
    shots = files()[2]; r = rng_of('seq%d' % j); pick = [s for s in (_try(shots[int(k)]) for k in r.choice(len(shots), 6, replace=False)) if s is not None]
    if not pick: return None
    bpm = r.uniform(70, 160); s = 60 / bpm / 4; y = np.zeros((N + 8 * SR, 2)); t = 0.02
    while t < 5.6:
        if r.random() < 0.4: h = pick[int(r.integers(len(pick)))][:3 * SR]; i = int(t * SR); y[i:i + len(h)] += h * r.uniform(0.4, 1.0)
        t += s
    return y[:N]
def k_wave(j):                                    # notes played from a single-cycle wave (a bare synth voice: nothing but the wave and an envelope)
    waves = files()[3]; r = rng_of('wave%d' % j); w = _try(waves[j % len(waves)])
    if w is None or len(w) < 16: return None
    w = w[:, 0]; w = w - w.mean(); y = np.zeros(N + SR); t = r.uniform(0.02, 0.2); base = r.choice([0.15, 0.2, 0.25, 0.4, 0.5]); root = r.integers(36, 60); sc = SCALES[r.integers(len(SCALES))]
    while t < 5.5:
        d = base * r.choice([1, 1, 2, 2, 3, 4]); f0 = 440 * 2 ** ((root + r.choice(sc) + 12 * r.integers(0, 2) - 69) / 12); L = int(min(d * r.uniform(0.5, 1.0), 1.5) * SR)
        ph = (np.arange(L) * f0 / SR) % 1.0; v = np.interp(ph * len(w), np.arange(len(w) + 1), np.r_[w, w[0]]); tt = np.arange(L) / SR
        env = np.minimum(1, tt / r.uniform(0.002, 0.05)) * np.exp(-tt * r.uniform(0.5, 8)) * np.minimum(1, (L - np.arange(L)) / (0.01 * SR)); i = int(t * SR); y[i:i + L] += v * env * r.uniform(0.4, 1); t += d
    y = ss.lfilter(*ss.butter(1, 16000 / (SR / 2)), y); return np.stack([y[:N], y[:N]], 1)

SENT = ["Good morning, this is the nine o'clock news from the capital.", "She sells sea shells by the sea shore, but the shells she sells are not hers.", "Two tickets to the last train, please, and a window seat if you have one.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?", "The weather will stay dry until the evening, then rain moves in from the west.", "Press the red button twice and wait for the green light to come on.",
        "My uncle has kept bees for forty years and has never once been stung.", "One, two, three, four, five, six, seven, eight, nine, ten.", "Is anybody there? Hello? I thought I heard somebody at the door.",
        "We apologise for the delay. The next service will depart from platform four.", "A long time ago, in a small village by the mountains, there lived an old baker.", "Remember to save your work before you close the window.",
        "Yes. No. Maybe. I really do not know what to tell you.", "The museum opens at ten and closes at half past five, except on Mondays.", "Slowly, very slowly, the big ship turned around and left the harbour."]
VOICES = ['en-us+m1', 'en-gb+f2', 'en-029', 'en-gb-scotland', 'en-gb-x-rp', 'en-us+m7', 'en-us+f5', 'it', 'nl', 'pt', 'pl', 'sv', 'da', 'cs', 'en-gb-x-gbclan', 'en-us+m4', 'en-gb+f1', 'en-us+f2', 'tr', 'fi']
PICO = ['en-US', 'en-GB', 'de-DE', 'fr-FR', 'es-ES', 'it-IT']
def k_tts(j):
    r = rng_of('tts%d' % j); w = os.path.join(TMPD, 't%d.wav' % os.getpid()); text = SENT[j % len(SENT)] + ' ' + SENT[(j * 4 + 7) % len(SENT)]
    if j % 3 == 2: subprocess.run(['pico2wave', '-l', PICO[(j // 3) % len(PICO)], '-w', w, text], check=True, capture_output=True)
    else: subprocess.run(['espeak-ng', '-v', VOICES[(j - j // 3) % len(VOICES)], '-s', str(int(r.integers(120, 200))), '-p', str(int(r.integers(20, 80))), '-w', w, text], check=True, capture_output=True)
    x = load(w); os.remove(w)
    if len(x) > N + SR and r.random() < 0.5: x = x[int(r.integers(0, len(x) - N)):]
    return x

def k_synth(j):
    r = rng_of('syn%d' % j); t = np.arange(N) / SR; kind = j % 8; st = lambda v: np.stack([v, v], 1)
    if kind == 0:   # plucked strings (a delay line with a damping loop)
        y = np.zeros(N + SR); tt = r.uniform(0.02, 0.2)
        while tt < 5.5:
            f0 = 55 * 2 ** (r.integers(0, 40) / 12); P = int(SR / f0); L = int(r.uniform(0.3, 1.5) * SR); buf = r.standard_normal(P); out = np.zeros(L); damp = r.uniform(0.990, 0.9995)
            for k in range(0, L, P):
                m = min(P, L - k); out[k:k + m] = buf[:m]; buf = damp * 0.5 * (buf + np.roll(buf, 1))
            i = int(tt * SR); y[i:i + L] += out * r.uniform(0.4, 1); tt += r.uniform(0.15, 0.9)
        return st(y[:N])
    if kind == 1:   # held chords of plain partials with slow attacks
        y = np.zeros(N); tt = 0.0
        while tt < 5.5:
            d = r.uniform(0.8, 2.5); L = int(min(d, 6 - tt) * SR); ts = np.arange(L) / SR; root = 110 * 2 ** (r.integers(0, 24) / 12); v = np.zeros(L)
            for iv in (0, r.choice([3, 4]), 7, 12):
                f0 = root * 2 ** (iv / 12); v += sum(np.sin(2 * np.pi * f0 * h * ts + r.uniform(0, 6.28)) / h ** r.uniform(1, 2) for h in range(1, int(r.integers(2, 9))))
            env = np.minimum(1, ts / r.uniform(0.05, 0.5)) * np.minimum(1, (L - np.arange(L)) / (0.1 * SR)); i = int(tt * SR); y[i:i + L] += v * env; tt += d
        return st(y)
    if kind == 2:   # beeps and bleeps: plain sines, steady or gliding, with gaps
        y = np.zeros(N); tt = r.uniform(0, 0.3)
        while tt < 5.7:
            L = int(r.uniform(0.04, 0.5) * SR); ts = np.arange(L) / SR; f0 = r.uniform(200, 3000); f1 = f0 * (2 ** r.uniform(-1, 1) if r.random() < 0.4 else 1); ph = 2 * np.pi * np.cumsum(np.geomspace(f0, f1, L)) / SR
            env = np.minimum(1, ts * 300) * np.minimum(1, (L - np.arange(L)) / (0.005 * SR)); i = int(tt * SR); m = min(L, N - i); y[i:i + m] += (np.sin(ph) * env)[:m]; tt += L / SR + r.uniform(0.02, 0.6)
        return st(y)
    if kind == 3:   # noise that swells and fades, band by band (wind, surf): slow and not periodic
        X = r.standard_normal(N); y = np.zeros(N)
        for _ in range(int(r.integers(2, 5))):
            lo = r.uniform(80, 3000); b = ss.lfilter(*ss.butter(2, [lo / (SR / 2), min(0.9, lo * r.uniform(1.5, 6) / (SR / 2))], 'band'), X)
            k = np.cumsum(r.standard_normal(N // 4800 + 2)); k = np.interp(np.arange(N), np.arange(len(k)) * 4800, k); y += b * 10 ** ((k - k.max()) * r.uniform(0.5, 3) / 20)
        return st(y)
    if kind == 4:   # two independent noises left and right through the same colouring (a wide, uncorrelated sound that is nobody's effect)
        f = np.maximum(np.fft.rfftfreq(N, 1 / SR), 20); c = f ** (-r.uniform(0, 2) / 2); a = np.fft.irfft(np.fft.rfft(r.standard_normal(N)) * c, N); b = np.fft.irfft(np.fft.rfft(r.standard_normal(N)) * c, N); m = r.uniform(0, 1)
        e = np.ones(N)
        for s0 in np.sort(r.integers(0, N - SR // 2, r.integers(0, 6))): L = int(r.uniform(0.1, 0.8) * SR); e[s0:s0 + L] *= r.uniform(0, 0.3)
        e = ss.lfilter([0.002], [1, -0.998], e); return np.stack([(a + m * b) * e, (b + m * a) * e], 1)
    if kind == 5:   # struck bars and bells: inharmonic partials, each with its own decay
        y = np.zeros(N + 2 * SR); tt = r.uniform(0.02, 0.3)
        while tt < 5.5:
            L = int(r.uniform(0.4, 2.0) * SR); ts = np.arange(L) / SR; f0 = 150 * 2 ** (r.integers(0, 36) / 12); v = sum(np.sin(2 * np.pi * f0 * q * ts) * np.exp(-ts * r.uniform(2, 15) * q ** 0.5) / q for q in (1, 2.76, 5.4, 8.93, 13.34)[:int(r.integers(2, 6))])
            i = int(tt * SR); y[i:i + L] += v * r.uniform(0.3, 1) * np.minimum(1, ts * 2000); tt += r.uniform(0.15, 1.0)
        return st(y[:N])
    if kind == 6:   # a voice-like buzz: a pulse train through three fixed resonances, with a drifting pitch and pauses
        f0 = r.uniform(80, 240) * 2 ** (0.2 * np.cumsum(r.standard_normal(N)) / np.sqrt(N) * 3); ph = np.cumsum(f0) / SR; x = np.diff(np.floor(ph), prepend=0.0) + 0.02 * r.standard_normal(N); y = np.zeros(N)
        for fc in (r.uniform(300, 900), r.uniform(900, 2400), r.uniform(2400, 3600)): y += ss.lfilter(*ss.iirpeak(fc / (SR / 2), r.uniform(4, 12)), x)
        e = np.ones(N)
        for s0 in np.sort(r.integers(0, N - SR // 2, r.integers(2, 7))): L = int(r.uniform(0.1, 0.5) * SR); e[s0:s0 + L] = 0
        return st(y * ss.lfilter([0.003], [1, -0.997], e))
    # kind 7: a tone that glides (portamento) with harmonics, steady level
    f = r.uniform(80, 500) * 2 ** np.interp(t, np.sort(r.uniform(0, 6, 8)), r.uniform(-1, 1, 8)); ph = 2 * np.pi * np.cumsum(f) / SR
    return st(sum(np.sin(h * ph + r.uniform(0, 6.28)) / h ** r.uniform(0.7, 2) for h in range(1, int(r.integers(2, 14)))))

KINDS = [('fluid-melody', k_fluid_mel, 44), ('fluid-chords', k_fluid_chd, 16), ('fluid-stabs', k_fluid_stab, 12), ('fluid-band', k_fluid_band, 24), ('sample-loop', k_loop, 30),
         ('sample-notes', k_notes, 14), ('sample-sequence', k_seq, 10), ('synth-voice', k_wave, 10), ('speech-synth', k_tts, 30), ('synthetic', k_synth, 24)]
INDEX = [(kind, fn, j) for kind, fn, n in KINDS for j in range(n)]
_o = np.random.default_rng(5).permutation(len(INDEX)); INDEX = [INDEX[i] for i in _o]      # mixed order, so that any first part of the list has all kinds
COUNT = len(INDEX)
def name_of(i): kind, fn, j = INDEX[i]; return 'x_%s_%02d' % (fn.__name__[2:], j)
def make(i):
    kind, fn, j = INDEX[i]; name = name_of(i)
    try: x = fn(j)
    except Exception as e: return name, kind, None
    return name, kind, (finish(x) if x is not None else None)

if __name__ == '__main__':
    import time, collections
    c = collections.Counter(); t0 = time.time()
    for i in (range(COUNT) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]):
        name, kind, x = make(i); c[(kind, x is not None)] += 1
        if x is None: print('none:', name)
        else: print('%-22s %-16s side/mid %6.1f dB  rms %.3f' % (name, kind, 10 * np.log10(((x[:, 0] - x[:, 1]) ** 2).mean() / ((x[:, 0] + x[:, 1]) ** 2).mean() + 1e-12), np.sqrt((x ** 2).mean())))
    print(dict(c), '%.1f s' % (time.time() - t0))
