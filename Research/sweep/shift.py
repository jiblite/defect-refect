# Where "onto" scores low: is the movement right but slightly off in time, or wrong? For every bench sound on which the engine
# finds the flanger: how much of the found line lies within 2.5 samples of the true one (measured with the dry sound known,
# truth_flanger.pkl: on the sound itself and on the other sound, the one "onto" is judged on), and the "onto" score with the
# movement shifted in time by up to +-160 ms (best shift), and with the true gain and feedback in place of the found ones.
#   python3 shift.py [case substring]
import sys, os, json, pickle, copy, numpy as np
os.chdir('/tmp/claude-0/v2/sweep'); sys.path.insert(0, '/tmp/claude-0/v2/sweep'); from lab import *
import sweep
sub = sys.argv[1] if len(sys.argv) > 1 else ''; T = pickle.load(open('truth_flanger.pkl', 'rb')); rows = []
SH = (0, 5, -5, 10, -10, 20, -20, 40, -40, 80, -80, 160, -160)
def near(f, t):
    m = (f['weight'] > 0.05) & (t['weight'] > 0.02); n = min(len(f['d']), len(t['d']))
    return float((np.abs(f['d'][:n] - t['d'][:n])[m[:n]] < 2.5).mean()) if m[:n].sum() > 5 else float('nan')
def shifted(F, ms):
    G = dict(F); G['ch'] = []
    for f in F['ch']:
        if f is None: G['ch'].append(None); continue
        g = dict(f); k = ms * 0.001 * sweep.SR / sweep.HOP; T_ = len(f['d']); g['d'] = np.interp(np.arange(T_) - k, np.arange(T_), f['d']); G['ch'].append(g)
    return G
for d in cases('flanger'):
    if sub not in d: continue
    cA = load(d, 'A'); cB = load(d, 'B')
    for c, o, tag, ot in ((cA, cB, 'A', 'B'), (cB, cA, 'B', 'A')):
        F = sweep.find(c['wet'])
        if F is None: continue
        tr = T[c['name']]; own = np.nanmean([near(f, tr[tag][k]) for k, f in enumerate(F['ch']) if f is not None]); oth = np.nanmean([near(f, tr[ot][k]) for k, f in enumerate(F['ch']) if f is not None])
        sc = {ms: score.removed(o['wet'], o['dry'], sweep.apply(o['dry'], shifted(F, ms), 1.0))[0] for ms in SH}; best = max(sc, key=sc.get)
        G = dict(F); G['ch'] = [None if f is None else dict(f, g=tr[ot][k]['g'], fb=tr[ot][k]['fb']) for k, f in enumerate(F['ch'])]
        sg = score.removed(o['wet'], o['dry'], sweep.apply(o['dry'], G, 1.0))[0]
        H = dict(F); H['ch'] = [None if f is None else dict(f, d=tr[ot][k]['d'][:len(f['d'])] if len(tr[ot][k]['d']) >= len(f['d']) else np.pad(tr[ot][k]['d'], (0, len(f['d']) - len(tr[ot][k]['d'])), mode='edge')) for k, f in enumerate(F['ch'])]
        sd = score.removed(o['wet'], o['dry'], sweep.apply(o['dry'], H, 1.0))[0]
        rows.append(dict(name=c['name'], which=tag, own=own, other=oth, onto=sc[0], best_ms=best, best=sc[best], true_gain=sg, true_line=sd))
        print('%-20s %s line within 2.5 samples of the true one: %3.0f %% (own sound) %3.0f %% (other sound) | onto %6.1f | best shift %+4d ms: %6.1f | with the true gain, feedback: %6.1f | with the true line of the other sound, found gain: %6.1f' % (
            c['name'], tag, 100 * own, 100 * oth, sc[0], best, sc[best], sg, sd), flush=True)
json.dump(rows, open('shift.json', 'w'))
