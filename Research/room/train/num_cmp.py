# Which answer should the room numbers be read from: the first look's or the second look's? Both against the true numbers
# of the scoreboard sounds (roomtruth_sb.npz), mid bands.   python3 num_cmp.py first second
import sys, glob, json, os
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, torch, common as cm, net, roomnum, predict
torch.set_num_threads(1)
A, B = sys.argv[1], sys.argv[2]; tr = np.load('/tmp/claude-0/v2/train/roomtruth_sb.npz', allow_pickle=True); names = list(tr['names']); mid = (cm.FC41 > 450) & (cm.FC41 < 7000); res = {A: [], A + '+' + B: []}
for f in sorted(glob.glob('/tmp/claude-0/v2/train/sb/*.npz')):
    z = np.load(f); ft = dict(LE=z['LE'].astype(np.float64), U64=z['U64'].astype(np.float64), U256=z['U256'].astype(np.float64)); i = names.index(os.path.basename(f)[:-4]); ok = tr['ok'][i] & mid
    for m in res:
        rn = roomnum.room_numbers(ft['LE'], predict.gains(ft, m))
        if rn is None or ok.sum() < 3: res[m].append((np.nan, np.nan, rn is not None)); continue
        res[m].append((float(np.median(np.log2(rn[1][ok] / tr['T60'][i][ok]))), float(np.median(rn[0][ok] - tr['kdb'][i][ok])), True))
for m, v in res.items():
    a = np.array(v, float); t = a[:, 0][np.isfinite(a[:, 0])]; k = a[:, 1][np.isfinite(a[:, 1])]
    print('%-24s room found on %3.0f%% | decay time: bias x%.2f, within 20%% on %2.0f%%, within 35%% on %2.0f%% | strength: bias %+.1f dB, median |error| %.1f dB, within 3 dB on %2.0f%%' % (m, 100 * np.mean(a[:, 2]), 2 ** np.median(t), 100 * np.mean(np.abs(t) < np.log2(1.2)), 100 * np.mean(np.abs(t) < np.log2(1.35)), np.median(k), np.median(np.abs(k)), 100 * np.mean(np.abs(k) < 3)))
