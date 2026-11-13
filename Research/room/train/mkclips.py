# Dry clips for TRAINING only (the scoreboard has its own clips; none of its test clips or their source files are used here).
# 48 kHz stereo, 6 s, FLAC 24 bit. Kinds as in the scoreboard corpus, many more of each, made with new random draws.
import numpy as np, soundfile as sf, scipy.signal as ss, subprocess, os, glob, json, pretty_midi, sys, hashlib
sys.path.insert(0, '/tmp/claude-0/v2/corpus')
SR = 48000; N = 6 * SR; OUT = '/tmp/claude-0/v2/train/dry/'; RAW = '/tmp/claude-0/v2/train/raw/'; os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)
SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'; sb = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
from multiprocessing import Pool
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
def save(name, x, kind):
    x = fit(x); x = x - x.mean(0); pk = np.abs(x).max()
    if pk < 1e-4 or (x ** 2).mean() < 1e-9 * pk * pk * 100: return None
    x = x * (0.5 / pk); f = np.ones(len(x)); k = int(0.01 * SR); f[:k] = np.linspace(0, 1, k); f[-k:] = np.linspace(1, 0, k)
    sf.write(OUT + name + '.flac', x * f[:, None], SR, format='FLAC', subtype='PCM_24'); return (name, kind)
def render_midi(pm, tag):
    mid = RAW + tag + '.mid'; wav = RAW + tag + '.wav'; pm.write(mid)
    subprocess.run(['fluidsynth', '-ni', '-F', wav, '-r', str(SR), '-g', '0.8', '-o', 'synth.reverb.active=0', '-o', 'synth.chorus.active=0', SF2, mid], capture_output=True, check=True)
    x = load(wav); os.remove(wav); os.remove(mid); return x
import corpus as C     # the note generators (melody, chords, drums, stabs, sequence); nothing is written by importing
def pan(inst, rng, wide=1.0):
    inst.control_changes.append(pretty_midi.ControlChange(10, int(np.clip(64 + wide * rng.uniform(-60, 60), 0, 127)), 0.0)); return inst
def rng_of(tag): return np.random.default_rng(int(hashlib.md5(('train' + tag).encode()).hexdigest(), 16) % 2 ** 32)
def job(j):
    kind, i = j; tag = '%s_%03d' % (kind, i); rng = rng_of(tag)
    if os.path.exists(OUT + 'tr_' + tag + '.flac'): return ('tr_' + tag, kind)
    try:
        if kind == 'fluid-melody':
            prog = int(rng.integers(0, 120)); lo = int(rng.choice([28, 36, 48, 55])); pm = C.melody(rng, prog, lo, lo + int(rng.integers(20, 34)))
            if rng.random() < 0.5: pan(pm.instruments[0], rng)
            return save('tr_' + tag, render_midi(pm, tag), kind)
        if kind == 'fluid-chords': return save('tr_' + tag, render_midi(C.chords(rng, int(rng.integers(0, 104))), tag), kind)
        if kind == 'fluid-drums': pm, bpm = C.drums(rng); return save('tr_' + tag, render_midi(pm, tag), kind)
        if kind == 'fluid-stabs': return save('tr_' + tag, render_midi(C.stabs(rng, int(rng.integers(0, 120))), tag), kind)
        if kind == 'fluid-band':
            pm, bpm = C.drums(rng) if rng.random() < 0.8 else (pretty_midi.PrettyMIDI(), 100)
            if rng.random() < 0.85: pm.instruments.append(pan(C.melody(rng, int(rng.choice([32, 33, 34, 35, 36, 38, 39])), 28, 48, legato=0.9).instruments[0], rng, 0.2))
            for _ in range(int(rng.integers(1, 3))): pm.instruments.append(pan(C.chords(rng, int(rng.integers(0, 96))).instruments[0], rng))
            if rng.random() < 0.6: pm.instruments.append(pan(C.melody(rng, int(rng.integers(0, 120)), 55, 84).instruments[0], rng))
            return save('tr_' + tag, render_midi(pm, tag), kind)
        if kind == 'sample-sequence':
            SP = '/usr/share/sonic-pi/samples/'; grp = list(rng.choice(['bd', 'sn', 'drum', 'elec', 'perc', 'tabla', 'glitch', 'misc', 'ambi', 'bass', 'vinyl'], int(rng.integers(1, 4)), replace=False))
            files = [f for g in grp for f in sorted(glob.glob(SP + g + '_*.flac'))]; shots = [load(f) for f in rng.choice(files, min(len(files), 10), replace=False)]
            shots = [s[:SR * 2] for s in shots]; return save('tr_' + tag, C.sequence(rng, shots), kind)
        if kind == 'sample-notes':
            LM = '/usr/share/lmms/samples/'; files = sorted(glob.glob(LM + 'instruments/*.ogg')) + sorted(glob.glob(LM + 'stringsnpads/*.ogg')) + sorted(glob.glob(LM + 'basses/*.ogg')) + sorted(glob.glob(LM + 'misc/*.ogg'))
            files = files[10:]                                                  # the first ten are the scoreboard's
            x = load(str(rng.choice(files))); y = np.zeros((N + 4 * SR, 2)); t = 0.05
            while t < 5.0:
                r = 2 ** (rng.integers(-7, 8) / 12); seg = x[:min(len(x), 2 * SR)]; h = ss.resample(seg, max(64, int(len(seg) / r)), axis=0); L = min(len(h), int(rng.uniform(0.3, 1.2) * SR))
                w = np.ones(L); w[-min(L, 2400):] = np.linspace(1, 0, min(L, 2400)); k = int(t * SR); y[k:k + L] += h[:L] * w[:, None]; t += rng.uniform(0.3, 1.0)
            return save('tr_' + tag, y[:N], kind)
        if kind == 'sample-loop':
            SP = '/usr/share/sonic-pi/samples/'; LM = '/usr/share/lmms/samples/'
            files = [f for f in sorted(glob.glob(SP + 'loop_*.flac')) + sorted(glob.glob(SP + 'guit_*.flac')) + sorted(glob.glob(SP + 'bass_*.flac'))[:4] + sorted(glob.glob(SP + 'mehackit_*.flac'))[:4] if sb.get('sp_' + os.path.basename(f)[:-5], {}).get('split') == 'dev']
            files += sorted(glob.glob(SP + 'mehackit_*.flac'))[4:] + sorted(glob.glob(SP + 'bass_*.flac'))[4:] + sorted(glob.glob(SP + 'ambi_*.flac')) + sorted(glob.glob(SP + 'arovane_*.flac')) + sorted(glob.glob(SP + 'tbd_*.flac')) + sorted(glob.glob(SP + 'hat_*.flac'))
            x = load(str(rng.choice(files))); r = rng.uniform(0.85, 1.18); x = ss.resample_poly(x, 1000, int(1000 * r), axis=0); x = fit(x, max(N, len(x)), loop=True); s0 = int(rng.integers(0, max(1, len(x) - N + 1)))
            return save('tr_' + tag, fit(x[s0:], loop=True), kind)
        if kind == 'speech-synth':
            words = ("the a one this that my your our some every quick slow bright dark cold warm old new red green little big river station window garden market engine doctor teacher morning evening summer winter "
                     "runs walks sings opens closes keeps takes gives finds hears watches carries under over behind between near after before with without and but because while slowly loudly never always again today tomorrow").split()
            text = '. '.join(' '.join(rng.choice(words, int(rng.integers(5, 11)))) for _ in range(3)) + '.'; w = RAW + tag + '.wav'
            if rng.random() < 0.7:
                v = str(rng.choice(['en-us', 'en-gb', 'de', 'fr-fr', 'es', 'it', 'nl', 'pl', 'pt', 'sv'])) + str(rng.choice(['', '+f1', '+f2', '+f3', '+f4', '+m1', '+m2', '+m3', '+m5', '+m7']))
                subprocess.run(['espeak-ng', '-v', v, '-s', str(int(rng.uniform(120, 190))), '-p', str(int(rng.uniform(20, 80))), '-w', w, text], check=True, capture_output=True)
            else: subprocess.run(['pico2wave', '-l', str(rng.choice(['en-US', 'en-GB', 'de-DE', 'fr-FR', 'es-ES', 'it-IT'])), '-w', w, text], check=True, capture_output=True)
            x = load(w); os.remove(w); s0 = int(rng.integers(0, max(1, len(x) - N // 2))); return save('tr_' + tag, x[s0:], kind)
        if kind == 'real-recording':
            H = os.environ.get('OWN_RECORDINGS', 'recordings/'); test = set(v['note'].split()[1] for v in sb.values() if v['kind'] == 'real-recording' and v['split'] == 'test')
            files = []
            for f in sorted(glob.glob(H + '*')):
                try: inf = sf.info(f)
                except Exception: continue
                if os.path.basename(f)[:8] not in test and inf.duration >= 2.0: files += [f] * max(1, int(inf.duration / 6))
            x = load(files[i % len(files)]); s0 = int(rng.integers(0, max(1, len(x) - N + 1))); return save('tr_' + tag, fit(x[s0:s0 + N]), kind)
        if kind == 'synthetic':
            t = np.arange(N) / SR; st = lambda v: np.stack([v, v], 1); m = int(rng.integers(0, 6))
            if m == 0:
                x = np.zeros(N)
                for s0 in np.sort(rng.integers(0, N - SR // 2, rng.integers(4, 14))):
                    L = int(rng.uniform(0.02, 0.5) * SR); x[s0:s0 + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / SR * rng.uniform(4, 80)) * rng.uniform(0.3, 1)
                x = ss.lfilter(*ss.butter(2, [rng.uniform(60, 400) / (SR / 2), rng.uniform(3000, 15000) / (SR / 2)], 'band'), x)
            elif m == 1:
                X = np.fft.rfft(rng.standard_normal(N)); f = np.maximum(np.fft.rfftfreq(N, 1 / SR), 20); x = np.fft.irfft(X / f ** (rng.uniform(0, 2) / 2), N)
                if rng.random() < 0.6: x *= 0.15 + (np.sin(2 * np.pi * rng.uniform(0.3, 4) * t + rng.uniform(0, 6)) > rng.uniform(-0.5, 0.5))          # gated on and off
            elif m == 2:
                f = rng.uniform(100, 600) * 2 ** (rng.uniform(0.3, 1.2) * np.sin(2 * np.pi * rng.uniform(0.1, 1.0) * t)); ph = 2 * np.pi * np.cumsum(f) / SR; x = sum(np.sin(h * ph) / h ** rng.uniform(0.5, 1.5) for h in range(1, int(rng.integers(3, 12))))
            elif m == 3:
                f = rng.uniform(20, 90) * 2 ** (rng.uniform(-1, 1.5) * np.clip(t / t[-1] * rng.uniform(0.5, 2), 0, 1)); ph = np.cumsum(f) / SR; x = ss.lfilter(*ss.butter(2, rng.uniform(800, 6000) / (SR / 2)), (np.mod(ph, 1) < 0.15).astype(float) + 0.3 * rng.standard_normal(N))
            elif m == 4:
                x = np.zeros(N)
                for s0 in range(0, N - SR, int(rng.uniform(0.25, 0.9) * SR)):
                    L = int(rng.uniform(0.1, 0.6) * SR); tt = np.arange(L) / SR; f0 = 110 * 2 ** (rng.integers(0, 36) / 12)
                    x[s0:s0 + L] += np.sin(2 * np.pi * f0 * tt + rng.uniform(0, 4) * np.sin(2 * np.pi * f0 * rng.integers(1, 4) * tt)) * np.minimum(1, tt * 200) * np.exp(-tt * rng.uniform(1, 14))
            else:
                x = np.zeros(N)                                                # clicks and short tone pips: the cleanest look at a room
                for s0 in np.sort(rng.integers(0, N - SR // 4, rng.integers(3, 16))):
                    L = int(rng.uniform(0.002, 0.06) * SR); tt = np.arange(L) / SR; x[s0:s0 + L] += np.sin(2 * np.pi * rng.uniform(200, 6000) * tt) * np.hanning(L) * rng.uniform(0.3, 1) if rng.random() < 0.6 else rng.standard_normal(L) * np.hanning(L)
            return save('tr_' + tag, st(x), kind)
    except Exception as ex:
        return ('FAILED ' + tag + ' ' + str(ex)[:160], kind)
PLAN = [('fluid-melody', 220), ('fluid-chords', 70), ('fluid-drums', 60), ('fluid-stabs', 40), ('fluid-band', 140), ('sample-sequence', 90), ('sample-notes', 50), ('sample-loop', 70), ('speech-synth', 80), ('real-recording', 130), ('synthetic', 110)]
if __name__ == '__main__':
    jobs = [(k, i) for k, n in PLAN for i in range(n)]; meta = {}
    with Pool(4) as p:
        for k, r in enumerate(p.imap_unordered(job, jobs, chunksize=4)):
            if r is None: continue
            if r[0].startswith('FAILED'): print(r[0], flush=True); continue
            meta[r[0]] = r[1]
            if k % 100 == 0: print(k, r, flush=True)
    json.dump(meta, open('/tmp/claude-0/v2/train/dry_meta.json', 'w'), indent=0)
    import collections; print(len(meta), collections.Counter(meta.values()))
