# Lab helpers shared by the experiments: case loading, true responses, FIR building, scoring.
import sys, glob, json, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, scipy.signal as ss, truth, score
SR = 48000
SETS = '/tmp/claude-0/v2/bench/sets/tone/'
WORK = '/tmp/claude-0/v2/tone/lab/work/'; os.makedirs(WORK, exist_ok=True)
def case_dirs(): return sorted(glob.glob(SETS + '*/'))
NT = 1 << 17
def true_H(c, n=NT):
    """Complex true response (2 channels, n/2+1 bins): channel o's response to a click on channel o (the tone effects have no crosstalk)."""
    return np.stack([np.fft.rfft(c['h'][o][:n, o], n) for o in range(2)])
def true_db(c, n=NT):
    H = true_H(c, n); return np.fft.rfftfreq(n, 1 / SR), 20 * np.log10(np.maximum(np.abs(H), 1e-7))
def cell_mean(f, v, grid):
    """Mean of v (given on the dense linear frequencies f) over each grid cell (cell edges half way to the neighbours, in the grid's own
    spacing); cells narrower than the dense spacing are interpolated."""
    g = np.asarray(grid, float); e = np.empty(len(g) + 1)
    if g[0] > 0 and np.allclose(np.diff(np.log(g[g > 0][:3])), np.diff(np.log(g[g > 0][:3]))[0]) and not np.allclose(np.diff(g[:3]), np.diff(g[:3])[0]):
        e[1:-1] = np.sqrt(g[:-1] * g[1:]); e[0] = g[0] ** 2 / e[1]; e[-1] = g[-1] ** 2 / e[-2]
    else:
        e[1:-1] = 0.5 * (g[:-1] + g[1:]); e[0] = g[0] - 0.5 * (g[1] - g[0]); e[-1] = g[-1] + 0.5 * (g[-1] - g[-2])
    cs = np.concatenate([[0.0], np.cumsum(v)]); df = f[1] - f[0]
    lo = np.clip(np.round(e[:-1] / df).astype(int), 0, len(f)); hi = np.clip(np.round(e[1:] / df).astype(int), 0, len(f))
    out = np.interp(g, f, v); ok = hi - lo >= 2
    out[ok] = (cs[hi[ok]] - cs[lo[ok]]) / (hi[ok] - lo[ok]); return out
_cache = {}
def load(d, which='A'):
    return truth.load(d, which)
def base_dw(name):
    """d(dry, wet) for A and d(B_wet, B_dry) for B, cached."""
    p = WORK + 'dw.json'
    if 'dw' not in _cache: _cache['dw'] = json.load(open(p)) if os.path.exists(p) else {}
    return _cache['dw'].get(name)
def make_dw():
    out = {}
    for d in case_dirs():
        a = load(d, 'A'); b = load(d, 'B')
        out[a['name']] = dict(A=score.distance(a['dry'], a['wet']), B=score.distance(b['wet'], b['dry']), split=a['meta']['split'], effect=a['meta']['effect'])
    json.dump(out, open(WORK + 'dw.json', 'w'), indent=1); return out
def table(rows, title='', keys=None):
    """rows: list of dict(name, effect, split, <metric>: value...). Prints medians / means / share below zero per split and per effect."""
    keys = keys or [k for k in rows[0] if k not in ('name', 'effect', 'split')]
    def line(label, rs):
        s = '%-22s n=%2d ' % (label, len(rs))
        for k in keys:
            v = np.array([r[k] for r in rs], float); s += '| %s med %6.1f mean %6.1f <0 %3.0f%% ' % (k, np.median(v), v.mean(), 100 * (v < 0).mean())
        print(s)
    if title: print('== ' + title)
    for sp in ('dev', 'test'):
        rs = [r for r in rows if r['split'] == sp]
        if rs: line('ALL ' + sp, rs)
    for e in sorted(set(r['effect'] for r in rows)):
        rs = [r for r in rows if r['effect'] == e]; line(e + ' (' + rs[0]['split'] + ')', rs)
if __name__ == '__main__':
    t0 = time.time(); make_dw(); print(time.time() - t0)
