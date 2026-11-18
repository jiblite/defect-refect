# More chorus cases from the outside tool, at settings the scoreboard's draws do not cover: an even mix of sound and copy
# (the tool's own default), copies louder than the sound, strong feedback, a slow and a faint chorus.
#   writes /dev/shm/chx/<case>_{A,B}_{dry,wet}.wav and cases.json
import sys, os, json, numpy as np, soundfile as sf, pedalboard as pb
SR = 48000; OUT = '/dev/shm/chx/'; CORPUS = '/tmp/claude-0/v2/corpus/'
def chorus(x, p, pre=1.5):
    k = int(pre * SR); xin = np.vstack([np.zeros((k, 2)), x]); return pb.Pedalboard([pb.Chorus(**p)])(xin.T.astype(np.float32), SR, reset=True).T.astype(np.float64)[k:]
def dry(name):
    x = sf.read(CORPUS + 'dry/' + name + '.wav', always_2d=True)[0].astype(np.float64); return x if x.shape[1] > 1 else np.repeat(x, 2, 1)
src = [json.load(open('/tmp/claude-0/v2/bench/sets/chorus/%s/meta.json' % n)) for n in ('cho_calf_00', 'cho_tap_00', 'cho_sox_00', 'cho_calf_01', 'cho_tap_01', 'cho_pedalboard_01')]
P = [dict(rate_hz=1.0, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.5),
     dict(rate_hz=0.5, depth=0.4, centre_delay_ms=12.0, feedback=0.0, mix=0.5),
     dict(rate_hz=0.8, depth=0.3, centre_delay_ms=10.0, feedback=0.0, mix=0.7),
     dict(rate_hz=0.6, depth=0.3, centre_delay_ms=8.0, feedback=0.4, mix=0.25),
     dict(rate_hz=0.3, depth=0.5, centre_delay_ms=20.0, feedback=0.0, mix=0.4),
     dict(rate_hz=1.5, depth=0.15, centre_delay_ms=15.0, feedback=0.0, mix=0.2)]
cases = {}
for i, (p, m) in enumerate(zip(P, src)):
    name = 'x%02d' % i; cases[name] = dict(params=p, A=m['A'], B=m['B'])
    for w in 'AB':
        x = dry(m[w])[:, :2]; y = chorus(x, p); g = min(1.0, 0.98 / max(np.abs(y).max(), 1e-9)); sf.write(OUT + '%s_%s_dry.wav' % (name, w), x, SR, subtype='FLOAT'); sf.write(OUT + '%s_%s_wet.wav' % (name, w), y * g, SR, subtype='FLOAT')
    print(name, p, flush=True)
json.dump(cases, open(OUT + 'cases.json', 'w'), indent=1)
