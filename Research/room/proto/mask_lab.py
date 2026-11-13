# Mask estimators tried against the truth at band level (fast). Proxy score: like "removed %", computed on band powers.
import sys, glob, os, time
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, numba, room, dereverb as dv
CA = '/tmp/claude-0/v2/proto/cache2/'; DT = dv.DT; FLOOR = 1e-3
def proxy(E, S, rho):
    """100 = the masked band powers equal the true direct part, 0 = no better than the wet sound."""
    top = E.max(); fl = top * 1e-6; LS = 10 * np.log10(np.maximum(S, fl)); LE = 10 * np.log10(np.maximum(E, fl))
    LO = 10 * np.log10(np.maximum(E * np.maximum(1.0 - rho, FLOOR), fl)); cap = lambda a: np.minimum(np.abs(a), 30.0)
    return 100.0 * (1.0 - cap(LO - LS).sum() / max(cap(LE - LS).sum(), 1e-9))
def parts(E, S, R, rho):
    """Gain reached in true tails (dB, lower is better) and gain lost where the direct sound leads (dB, 0 is best)."""
    ft = R / np.maximum(S + R, 1e-30); live = E > E.max(0, keepdims=True) * 1e-5; g = 10 * np.log10(np.maximum(1.0 - rho, FLOOR)); gt = 10 * np.log10(np.maximum(1.0 - ft, FLOOR))
    tl = live & (ft > 0.9); dr = live & (ft < 0.3)
    return (g[tl].mean() if tl.sum() else np.nan), ((g - gt)[dr].mean() if dr.sum() else np.nan)
def model_rho(E, tau, kappa, beta=1.0):
    R = np.stack([room.reverb_track(E[:, b].astype(np.float64), tau[b], beta * kappa[b] / tau[b], DT) for b in range(E.shape[1])], 1); return R / np.maximum(E, 1e-30)
@numba.njit(cache=True)
def hmm_presence(gdb, sig_t, lo_d, hi_d, stay):
    """Probability that direct sound is present in each frame of one band, from g = level above the predicted reverb (dB).
    Two states: 'all tail' (g around 0 with spread sig_t) and 'sound present' (g anywhere between lo_d and hi_d); they
    tend to last (stay). Uses the whole file, forwards and backwards."""
    T = len(gdb); a = np.zeros((T, 2)); b = np.zeros((T, 2)); e = np.zeros((T, 2))
    for t in range(T):
        g = gdb[t]; e[t, 0] = np.exp(-0.5 * (g / sig_t) ** 2) / (sig_t * 2.5066) + 1e-6
        e[t, 1] = (1.0 / (hi_d - lo_d) if (g > lo_d and g < hi_d) else 0.0) + 1e-6
    a[0, 0] = 0.5 * e[0, 0]; a[0, 1] = 0.5 * e[0, 1]; s = a[0, 0] + a[0, 1]; a[0] /= s
    for t in range(1, T):
        a[t, 0] = (a[t - 1, 0] * stay + a[t - 1, 1] * (1 - stay)) * e[t, 0]; a[t, 1] = (a[t - 1, 1] * stay + a[t - 1, 0] * (1 - stay)) * e[t, 1]; s = a[t, 0] + a[t, 1]; a[t] /= s
    b[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        b[t, 0] = stay * e[t + 1, 0] * b[t + 1, 0] + (1 - stay) * e[t + 1, 1] * b[t + 1, 1]; b[t, 1] = stay * e[t + 1, 1] * b[t + 1, 1] + (1 - stay) * e[t + 1, 0] * b[t + 1, 0]; s = b[t, 0] + b[t, 1]; b[t] /= s
    p = a[:, 1] * b[:, 1]; return p / (p + a[:, 0] * b[:, 0] + 1e-30)
def load():
    out = []
    for f in sorted(glob.glob(CA + '*.npz')):
        z = dict(np.load(f)); z['name'] = os.path.basename(f)[:-4]
        if not (np.isfinite(z['tau']).all() and np.isfinite(z['kappa']).all()):
            tm = np.nanmedian(z['tau']); km = np.nanmedian(z['kappa'])
            if not np.isfinite(tm) or not np.isfinite(km): continue
            z['tau'] = np.where(np.isfinite(z['tau']), z['tau'], tm); z['kappa'] = np.where(np.isfinite(z['kappa']), z['kappa'], km)
        for k in ('E', 'S', 'R'): z[k] = z[k].astype(np.float64)
        out.append(z)
    return out
def table(cases, ests):
    mid = (room.FC > 450) & (room.FC < 7000); rows = []
    for z in cases:
        E, S, R = z['E'], z['S'], z['R']; sep = 10 * np.log10(np.median(z['u_rev'][mid]) / max(np.median(z['u_src'][mid]), 1e-6)); r = [sep, -10 * np.log10(np.mean(z['kappa'][mid]))]
        for k, fn in ests.items():
            rho = np.clip(fn(z), 0.0, 1.0); a, b = parts(E, S, R, rho); r += [proxy(E, S, rho), a, b]
        rows.append(r)
    a = np.array(rows); keys = list(ests.keys())
    for lab, sel in [('all', np.ones(len(a), bool)), ('stereo picture usable (reverb 8 dB wider than the dry sound)', a[:, 0] >= 8), ('not usable', a[:, 0] < 8), ('light (DRR >= 5)', a[:, 1] >= 5), ('wet (DRR < -2)', a[:, 1] < -2)]:
        print('%s  n=%d' % (lab, sel.sum()))
        for i, k in enumerate(keys): print('    %-34s score: median %5.1f mean %5.1f, below zero in %2.0f%% | tails turned down %5.1f dB | direct sound lost %5.1f dB' % (k, np.nanmedian(a[sel, 2 + 3 * i]), np.nanmean(a[sel, 2 + 3 * i]), 100 * np.mean(a[sel, 2 + 3 * i] < 0), np.nanmean(a[sel, 3 + 3 * i]), np.nanmean(a[sel, 4 + 3 * i])))
if __name__ == '__main__':
    t0 = time.time(); cases = load(); print(len(cases), 'cases')
    ests = {'true fraction': lambda z: z['R'] / np.maximum(z['S'] + z['R'], 1e-30), 'model': lambda z: model_rho(z['E'], z['tau'], z['kappa'])}
    for v in ('p064', 'p128', 'p256', 'b128', 'b256', 'b256w'):
        ests['stereo ' + v + ' /u_rev'] = (lambda v: lambda z: z['D_' + v] / np.maximum(z['P_' + v] * z['u_rev'][None], 1e-30))(v)
    for v in ('p128', 'b128', 'b256'):
        ests['stereo ' + v + ' two-point'] = (lambda v: lambda z: (z['D_' + v] / np.maximum(z['P_' + v], 1e-30) - z['u_src'][None]) / np.maximum(z['u_rev'] - z['u_src'], 1e-3)[None])(v)
    table(cases, ests); print('%.0f s' % (time.time() - t0))
