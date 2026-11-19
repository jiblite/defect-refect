# Ring modulations whose tone has a round pitch (so that its cycle is a whole number of samples and every cycle meets
# the samples at the same places), on clips of the corpus: does the fit end at the true tone?
#   python3 rf1.py [ladder]      prints per case: pitch and phase against the truth, the win against the win of a fit started at the truth
import sys, os, json, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import ring as E
def plain(x, f, rho, ph):
    t = np.arange(len(x)); g = (rho + np.sin(2 * np.pi * f / SR * t + ph)) / np.sqrt(rho * rho + 0.5); return x * g[:, None]
def cases(nmax=40):
    cm = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); clips = [c for c in sorted(cm) if not c.startswith('real_')]; rng = np.random.default_rng(11); out = []
    for i in range(nmax):
        f = float(rng.choice([50, 100, 120, 150, 200, 240, 250, 300, 400, 480, 500, 600, 750, 800, 960, 1000, 1200, 1500, 1600, 2000, 2400, 3000, 4000, 440, 441, 1627.118644]))
        ph = float(rng.choice([0.0, 0.0, rng.uniform(0, 6.283), rng.uniform(0, 6.283), 2 * np.pi * f / SR * float(rng.choice([0.01, 0.02, 0.04, 0.1, 0.3]))]))      # zeros on the samples, anywhere, or a hair beside them
        rho = float(rng.choice([0.0, 0.0, 0.0, rng.uniform(0.1, 0.9)])); out.append((str(rng.choice(clips)), f, rho, ph))
    return out
def study(y, f0, rho0, ph0, find):
    n = len(y); r = find(y); mid = 0.5 * n
    if 'f' not in r: return dict(found=False, measured=False)
    pw = E.power(y); W = E.window(r['f']); pn, ok = E.local(pw, W); t = np.arange(n)[ok]; p = pn[t]; half = 0.5 * n; tau = t - mid; tp = tau / half; flat = float(np.log(p.mean()))
    w0 = E.TWO_PI * f0 / SR; ut, Jt, k = E.settle(p, tau, tp, w0, np.array([w0 * mid + ph0, 0.0, rho0, np.log(1e-3)]))
    dph = float(np.angle(np.exp(1j * ((r['w'] * mid + r['phase']) - (w0 * mid + ph0)))))
    if abs(abs(dph) - np.pi) < 0.5 and r['rho'] < 0.05 and rho0 == 0: dph = float(np.angle(np.exp(1j * (dph - np.pi))))       # rho = 0: the same gain half a cycle on, sign apart
    return dict(found=bool(r['found']), measured=True, df=r['f'] - f0, dph=dph, rho=r['rho'], eps=r['eps'], win=r['win'], win_true=flat - Jt, passes=r['passes'], x=r.get('x'))
if __name__ == '__main__':
    bad = 0; rows = []
    for clip, f, rho, ph in cases(int(os.environ.get('N', 40))):
        x = truth.rd('/tmp/claude-0/v2/corpus/dry/' + clip + '.wav'); y = plain(x, f, rho, ph); t0 = time.time(); s = study(y, f, rho, ph, E.find); dt = time.time() - t0
        if not s['measured']: print('%-24s f %9.3f rho %.2f phase %.4f: NOT MEASURED' % (clip, f, rho, ph)); bad += 1; continue
        k = float((s['x'] * x).sum() / (x ** 2).sum()) if s['x'] is not None else 1.0; left = db(s['x'] - k * x, k * x) if s['x'] is not None else 0.0; off = s['win_true'] - s['win'] > 1e-4; bad += off or not s['found']; rows.append((s['win_true'] - s['win'], left))
        print('%-24s f %9.3f rho %.2f phase %.4f: found %d pitch %+.6f Hz phase %+.5f rho %.4f floor %.1e | win %.5f, from the truth %.5f %s| left %6.1f dB | %d passes' % (clip, f, rho, ph, s['found'], s['df'], s['dph'], s['rho'], s['eps'], s['win'], s['win_true'], 'LOWER ' if off else '', left, s['passes']), flush=True)
    print('%d cases, %d not at the best fit or not found; left beside the dry sound: median %.1f dB, worst %.1f dB' % (len(rows), bad, np.median([r[1] for r in rows]), max(r[1] for r in rows)))
