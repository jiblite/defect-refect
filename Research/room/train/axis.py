# Taking the reverb out with left and right looked at separately. Per band the sound itself has a main direction between
# the two channels (a centred voice: left = right) and the room usually has none (left and right tails differ). So the
# part of every band and moment that lies ACROSS the sound's direction is mostly room even while the sound is playing, and
# the part ALONG it holds less room than the band as a whole. Each of the two parts gets its own share of direct sound,
# worked out from the same band-level answer g and from how the sound's and the room's power split between them.
import sys
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, common as cm, render
_idx = np.clip(np.searchsorted(cm.EDGES, np.arange(cm.NF // 2 + 1), side='right') - 1, 0, cm.NB - 1)       # the band each FFT bin belongs to
def spatial(X, g_db, floor=50.0, sure=None):
    """Per band: u (2,) the unit direction of the sound itself between left and right, and the shares of the sound's power
    (dp along u, dq across) and of the room's power (rp, rq). Bands that cannot be told keep dp = dq = rp = rq = 0.5,
    which gives the plain rendering. From a fit of every moment's left/right power and cross power as (direct share) x
    (the sound's pattern) + (room share) x (the room's pattern)."""
    C = [(np.abs(X[0]) ** 2) @ cm.BM.T, (np.abs(X[1]) ** 2) @ cm.BM.T, (X[0] * np.conj(X[1])) @ cm.BM.T]; E = C[0] + C[1]; LE = 10 * np.log10(E + 1e-30); G = 10 ** (g_db / 10.0); a = G * E; c = (1.0 - G) * E
    live = LE > LE.max() - floor; u = np.tile(np.array([1.0, 0.0], complex), (cm.NB, 1)); sh = np.full((cm.NB, 4), 0.5)
    for b in range(cm.NB):
        m = live[:, b]
        if m.sum() < 20: continue
        aa = a[m, b]; cc = c[m, b]; Saa = (aa * aa).sum(); Scc = (cc * cc).sum(); Sac = (aa * cc).sum(); det = Saa * Scc - Sac * Sac
        if not (Saa > 0 and Scc > 0 and det > 1e-4 * Saa * Scc): continue
        Bd = []; Br = []
        for Ck in C: ra = (aa * Ck[m, b]).sum(); rc = (cc * Ck[m, b]).sum(); Bd.append((Scc * ra - Sac * rc) / det); Br.append((Saa * rc - Sac * ra) / det)
        def psd(B):
            M = np.array([[B[0].real, B[2]], [np.conj(B[2]), B[1].real]]); w, V = np.linalg.eigh(M); w = np.maximum(w, 0.0); return (w / w.sum(), V) if w.sum() > 0 else None
        d = psd(Bd); r = psd(Br)
        if d is None or r is None: continue
        ub = d[1][:, 1]; vb = np.array([-np.conj(ub[1]), np.conj(ub[0])]); Mr = (r[1] * r[0][None, :]) @ np.conj(r[1].T); rp = float(np.real(np.conj(ub) @ Mr @ ub)); rq = float(np.real(np.conj(vb) @ Mr @ vb))
        u[b] = ub; sh[b] = [d[0][1], d[0][0], rp / max(rp + rq, 1e-12), rq / max(rp + rq, 1e-12)]
    return u, sh
def shares(g_db, sh):
    """The direct share along and across the sound's direction, per bin: (T, bins) each."""
    G = 10 ** (render.to_bins(g_db) / 10.0); dp, dq, rp, rq = (sh[_idx, k][None, :] for k in range(4))
    return G * dp / (G * dp + (1.0 - G) * rp + 1e-12), G * dq / (G * dq + (1.0 - G) * rq + 1e-12)
def render_axis(wet, g_db, s=0.0, floor_db=-30.0, sp=None):
    """As render.render(), with the two directions treated separately. s: 0 = reverb out, 1 = untouched, 2 = reverb doubled."""
    X = cm.fwd(wet); u, sh = sp if sp is not None else spatial(X, g_db); Gp, Gq = shares(g_db, sh); fl = 10 ** (floor_db / 10.0)
    if s <= 1.0: gp = np.maximum(Gp, fl) ** ((1.0 - s) / 2.0); gq = np.maximum(Gq, fl) ** ((1.0 - s) / 2.0)
    else: gp = np.sqrt(1.0 + (s * s - 1.0) * (1.0 - Gp)); gq = np.sqrt(1.0 + (s * s - 1.0) * (1.0 - Gq))
    u0 = u[_idx, 0][None, :]; u1 = u[_idx, 1][None, :]; p = (np.conj(u0) * X[0] + np.conj(u1) * X[1]) * gp; q = (-u1 * X[0] + u0 * X[1]) * gq
    return cm.inv(np.stack([u0 * p - np.conj(u1) * q, u1 * p + np.conj(u0) * q]), len(wet))
