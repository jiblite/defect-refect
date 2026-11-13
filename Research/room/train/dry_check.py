# Dry sounds must not get a room: the reader on the scoreboard's dry clips (no effect at all).
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, soundfile as sf, common as cm, engine
corp = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); MODEL = os.environ.get('MODEL', 'unet_b9'); rows = []
for nm, m in sorted(corp.items()):
    x = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % nm, always_2d=True)[0].astype(np.float64); x = np.vstack([x, np.zeros((3 * 48000, 2))]); an = engine.analyse(x, MODEL); mid = (cm.FC41 > 450) & (cm.FC41 < 7000)
    g = an['g']; live = an['LE'] > an['LE'].max() - 50; rows.append((m['split'], m['kind'], an['found'], float(np.median(an['kdb'][mid])) if an['found'] else -99.0, float(np.median(an['T60'][mid])) if an['found'] else 0.0, float(g[live].mean()), float((g[live] < -10).mean()), float(an['ev'].sum()) if an['found'] else 0.0))
json.dump(rows, open('/tmp/claude-0/v2/train/dry_check_%s.json' % MODEL, 'w'))
for sp in ('dev', 'test'):
    r = [x for x in rows if x[0] == sp]; k = np.array([x[3] for x in r]); print('%-5s n=%3d | a room is reported on %2.0f%% | strength above -15 dB: %2.0f%%, above -10 dB: %2.0f%%, above -5 dB: %2.0f%% | mean gain %.2f dB, cells under -10 dB: %.1f%%' % (sp, len(r), 100 * np.mean([x[2] for x in r]), 100 * np.mean(k > -15), 100 * np.mean(k > -10), 100 * np.mean(k > -5), np.mean([x[5] for x in r]), 100 * np.mean([x[6] for x in r])))
kinds = sorted(set(x[1] for x in rows))
for kd in kinds:
    r = [x for x in rows if x[1] == kd]; k = np.array([x[3] for x in r]); print('  %-16s n=%3d | reported %3.0f%% | above -10 dB: %3.0f%% | median strength where reported %.1f dB, decay time %.2f s' % (kd, len(r), 100 * np.mean([x[2] for x in r]), 100 * np.mean(k > -10), np.median(k[k > -90]) if (k > -90).any() else np.nan, np.median([x[4] for x in r if x[2]]) if any(x[2] for x in r) else np.nan))
