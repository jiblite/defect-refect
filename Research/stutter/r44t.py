# The two engines on sounds that are not at 48 kHz (the app reads such a file at its own rate): 16 clips each at
# 44.1 kHz and at 96 kHz, a stutter made there (slices of whole samples, no fades) and a bare crusher made there.
# What is read against what was made.   python3 r44t.py
import sys, os, json, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
from scipy.signal import resample_poly
meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); ids = sorted(meta); rng = np.random.default_rng(47)
pick = [ids[i] for i in rng.choice(len(ids), 16, replace=False)]
tmp = tempfile.mkdtemp(dir='/dev/shm')
def stut(x, events):
    o = x.copy()
    for s0, L, end in events:
        for t in range(s0 + L, min(end, len(x)), L): k = min(L, end - t, len(x) - t); o[t:t + k] = x[s0:s0 + k]
    return o
for sr, (up, dn) in ((44100, (147, 160)), (96000, (2, 1))):
    okS = okC = foundS = foundC = made = right = 0; rows = []
    for cid in pick:
        x = resample_poly(truth.rd('/tmp/claude-0/v2/corpus/dry/' + cid + '.wav'), up, dn, axis=0); n = len(x)
        L = int(rng.uniform(0.03, 0.25) * sr); reps = int(rng.integers(2, 7)); pos = int(0.2 * sr); ev = []
        while pos + L * reps < n:
            if rng.random() < 0.4: ev.append((pos, L, pos + reps * L)); pos += reps * L
            else: pos += L
        live = [e for e in ev if np.abs(x[e[0]:e[0] + e[1]]).max() > 1e-3]
        sf.write(tmp + '/s.wav', stut(x, ev), sr, subtype='PCM_24')
        J = json.loads(subprocess.run(['/tmp/claude-0/drstutter_b', 'find', tmp + '/s.wav'], capture_output=True, text=True).stdout.strip().splitlines()[-1])
        got = set(map(tuple, J['events'])); foundS += J['found']; made += len(live); right += len(got & set(ev)); extra = len(got - set(ev))
        bits = int(rng.integers(4, 12)); hold = int(rng.integers(1, 9)); st = 2.0 / 2 ** bits; q = np.round(x / st) * st
        if hold > 1: q = q[(np.arange(n) // hold) * hold]
        sf.write(tmp + '/c.wav', q, sr, subtype='PCM_24')
        K = json.loads(subprocess.run(['/tmp/claude-0/drcrush_b', 'find', tmp + '/c.wav'], capture_output=True, text=True).stdout.strip().splitlines()[-1])
        foundC += K['found']; gridOk = K['found'] and all(g is None or abs(g['bits'] - bits) < 0.01 for g in K['grids']) and any(g is not None for g in K['grids']); holdOk = K['found'] and abs(K['P'] - hold) < 1e-6
        okC += bool(gridOk and holdOk); rows.append((cid if not cid.startswith('real_') else 'real_XX', len(live), len(got & set(ev)), extra, bits, hold, K.get('bits'), K.get('P'), K.get('kind')))
    print('%d Hz: stutter found on %d of 16 sounds; of %d events with a sounding slice, %d read right to the sample. Crusher found on %d of 16, grid and hold right on %d' % (sr, foundS, made, right, foundC, okC))
    for r in rows:
        if r[1] != r[2] or r[3] or not (r[6] is not None and abs(r[6] - r[4]) < 0.01 and abs(r[7] - r[5]) < 1e-6): print('     ', r)
for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
