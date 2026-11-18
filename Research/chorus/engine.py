# The chorus engine's prototype, end to end: the voices of a chorus found on a sound (find), taken out of it or changed
# (apply with own=True), or put onto another sound (own=False).
#   find(y)                      y (n, 2) -> dict(found, sure, voices [per side: list of voices with their loudness per range], n)
#   apply(y, F, s, speed, depth, own)
#       own=True : y is the sound the chorus was found on; s = -1 takes it out, 0 leaves the sound as it is, +1 doubles it
#       own=False: y is another sound; s = 0 nothing, 1 the chorus as found, 2 double
#       speed, depth: the voices' movement faster / wider (1 = as found)
#   python3 engine.py [case ...]   the scoreboard's chorus cases: taken out (against the dry sound) and put onto the other sound (against its wet sound)
import sys, json, time, numpy as np, numba
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
import chorus as C, inv, gains as Gn, prim
SR = 48000
CENTRE = np.sqrt(np.concatenate([[75.0], Gn.EDGES]) * np.concatenate([Gn.EDGES, [19200.0]]))      # the middle of each range, Hz
MID = (3, 4, 5)                                              # the ranges that say most about a voice's loudness: 600 Hz to 4.8 kHz
NEED = 0.25; LOUD = 0.08                                     # a voice must show in this share of the sounding frames, and be this loud in the middle ranges

def settle(G, V):
    """In a low range a voice's delay swings through less than one wave: there the sound's own steady tones are taken for the
    voice (or hide it), and the measure cannot be trusted. Such a range leans on the range above it. (Tried: leaving the
    lowest range alone, because one of the four outside tools thins its copies under 100 Hz. Its sounds gained 6 points, the
    sounds of the tools that do not lost 10.)"""
    G = G.copy()
    for v, q in enumerate(V):
        swing = 2.0 * abs(q['half']) / C.FS
        for b in range(Gn.NB - 2, -1, -1): w = min(1.0, CENTRE[b] * swing); G[b, v] = w * G[b, v] + (1.0 - w) * G[b + 1, v]
    return G

def loudness(y1, V, D, rounds=6):
    B = prim.bands(y1); G = np.zeros((Gn.NB, len(D)))
    for r in range(rounds):
        x = Gn.remove(y1, D, G, B); G = Gn.capped(settle(G + Gn.left(x, D), V))
    return G, Gn.remove(y1, D, G, B)

def spiky(x):
    """How peaked the sound's waveform is (of its sample to sample changes): taking a chorus out the right way makes it more peaked, never less."""
    z = np.diff(x); m2 = (z ** 2).mean(); return float((z ** 4).mean() / max(m2 * m2, 1e-30) - 3.0)

def curves(V, n, speed=1.0, depth=1.0, n0=None):
    """The voices' delays (samples at 48 kHz) for n samples. Faster or slower: the movement turns about the middle of the sound it was found on."""
    if not V: return np.zeros((0, n))
    t = np.arange(n) / (C.HOP * C.DEC); tm = (n0 if n0 is not None else n) / (C.HOP * C.DEC) / 2.0; out = []
    for v in V:
        q = dict(v); q['half'] = v['half'] * depth; q['rate'] = v['rate'] * speed; q['phase'] = v['phase'] + v['rate'] * tm / C.FPS * (1.0 - speed)
        d = C.curve(q, t)[0] * C.DEC; out.append(np.maximum(d, 8.0))
    return np.stack(out)

def clear(v):
    """How clear a voice is, 0 .. 1: seen in most frames, loud enough, there all the way round its movement, and well gone once taken out."""
    return v['on'] * min(1.0, v['loud'] / 0.2) * min(1.0, v['least'] / 0.35) * min(1.0, (1.0 - v['after'] / max(GONE * v['before'], 1e-12)) / 0.3)

@numba.njit(cache=True)
def _read(x, p):
    """x read at the places p (not whole)."""
    r = np.zeros(len(p))
    for i in range(len(p)): r[i] = inv._lag4(x, p[i])
    return r

def advance(V, n):
    """How far after sample m its copy is heard: a(m) = d(m + a(m)), for each voice."""
    t = np.arange(n, dtype=float); per = C.HOP * C.DEC; out = []
    for v in V:
        a = np.maximum(C.curve(v, t / per)[0] * C.DEC, 8.0)
        for _ in range(4): a = np.maximum(C.curve(v, (t + a) / per)[0] * C.DEC, 8.0)
        out.append(a)
    return np.stack(out)

def backwards(y1, V, H=None):
    """One copy that is LOUDER than the sound itself: y = x + g x(n - d), g over 1. Taken out the usual way, the louder part
    is kept and the sound itself removed in its place. Told from the copy's side the same sound is y = v + h v(n + a(n)):
    v the copy, h = 1 / g under 1, the sound itself an EARLIER, quieter copy of it. That is the usual case with time running
    backwards: so the sound is turned round, the loudness h read and v taken out as always, and turned back; the sound
    itself is then v read a(n) later and h times as loud. Returns h per range (NB, 1), the sound without the copy, and per
    range the copy brought back to its sound's place (u_b(m) = v_b(m + a(m)) = g_b x_b(m))."""
    n = len(y1); A = advance(V, n); yr = np.ascontiguousarray(y1[::-1]); Dr = np.ascontiguousarray(A[:, ::-1])
    if H is None: H, _ = loudness(yr, V, Dr)
    B = prim.bands(yr); t = np.arange(n, dtype=float) + A[0]; U = [_read(np.ascontiguousarray(inv._inverse(B[b], Dr, np.ascontiguousarray(H[b]))[::-1]), t) for b in range(Gn.NB)]
    return H, sum(H[b, 0] * U[b] for b in range(Gn.NB)), U

def lean(B, d):
    """Which comes first at a sharp event, the weaker or the stronger of the two? With e the size of the sound (range by
    range, from 150 Hz up) and ed its size read along the voice's curve: log(sum e^2 ed / sum e ed^2). Under 0: the later
    one (the copy) is the weaker, as in a usual chorus. Over 0: the copy is the louder. About 0 for a sound without sharp events."""
    A = 0.0; Bq = 0.0
    for b in range(1, Gn.NB): e = np.abs(B[b]); ed = np.abs(Gn._along(B[b], d)); A += float((e * e * ed).sum()); Bq += float((e * ed * ed).sum())
    return float(np.log(max(A, 1e-300) / max(Bq, 1e-300)))

def uneven(x):
    """How unevenly the sound's strength is spread in time, the ranges from 150 Hz up together (their fourth powers' means over their squares' means squared)."""
    B = prim.bands(x); return float(sum((B[b] ** 4).mean() for b in range(1, Gn.NB)) / max(sum((B[b] ** 2).mean() ** 2 for b in range(1, Gn.NB)), 1e-300))

UNSURE = -0.03                                               # a single voice whose lean is over this is also taken out the other way round, and the better way is kept
GONE = 0.45                                                  # a real voice is gone after taking it out: what is left along its curve is under this share of what was there
FLAT = 0.85                                                  # taking a chorus out never makes the waveform less peaked; if it falls under this share, the copies were louder than the sound itself (which this way of taking out cannot undo)
def find(y, say=None):
    n = len(y); sides = []; notes = []
    for ch in range(2):
        A, act = C.picture(y[:, ch]); T = len(A); P = C.peaks(A, act); info = []; V = C.voices(P, T, need=NEED, info=info); del A
        for v in info:
            if v['on'] >= NEED and (v['least'] < C.LEAST or v['twin'] >= C.TWIN): notes.append('side %d: no voice, %s: %s %.4f Hz %.3f +- %.3f ms on %.2f, least %.2f, twins %.2f' % (ch, 'the sound\'s own pitch' if v['twin'] >= C.TWIN else 'not there all the way round', 'tri' if v['kind'] else 'sin', v['rate'], v['mid'] / C.FS * 1e3, abs(v['half']) / C.FS * 1e3, v['on'], v['least'], v['twin']))
        while V:
            D = curves(V, n); G, xh = loudness(y[:, ch], V, D); keep = []; ahead = False; rides = None
            if len(V) == 1:                                            # one copy: is it the louder of the two? Then it is taken out the other way round
                L = lean(prim.bands(y[:, ch]), D[0]); V[0]['lean'] = L
                if L > UNSURE:
                    H, xo, _ = backwards(y[:, ch], V); h0, a0 = C.ride(y[:, ch], V[0]); hu, _ = C.ride(xh, V[0]); ho, _ = C.ride(xo, V[0]); bu = float(hu[a0].mean()); bo = float(ho[a0].mean()); pu = spiky(xh); po = spiky(xo)
                    votes = int(L > 0.02) + int(bo < 0.8 * bu) + int(pu > 1.0 and po > 1.05 * pu) + int(uneven(xo) > 1.05 * uneven(xh)); rides = (float(h0[a0].mean()), bu)
                    notes.append('side %d: one voice, lean %+.3f: the other way round gets %d of 4 votes (along %.3f against %.3f, peaked %.1f against %.1f)' % (ch, L, votes, bo, bu, po, pu))
                    k0 = spiky(y[:, ch]); ok = [float(np.mean(g[list(MID), 0])) >= LOUD and b < GONE * rides[0] and not (k0 > 1.0 and p < FLAT * k0) for g, b, p in ((G, bu, pu), (H, bo, po))]      # would the result stand, the usual way and the other way round?
                    if votes >= 3 or (ok[1] and not ok[0]): ahead = True; G = H; xh = xo; rides = (rides[0], bo)      # ... also when the usual way fails its tests and the other way passes them
            for i, v in enumerate(V):
                v['gain'] = [float(g) for g in G[:, i]]; v['ahead'] = ahead; v['loud'] = float(np.mean(G[list(MID), i]))
                if rides is None: h0, a0 = C.ride(y[:, ch], v); h1, a1 = C.ride(xh, v); v['before'] = float(h0[a0].mean()); v['after'] = float(h1[a0].mean())
                else: v['before'], v['after'] = rides
                if v['loud'] >= LOUD and v['after'] < GONE * v['before']: keep.append(v)
                else: notes.append('side %d: dropped %s %.4f Hz %.3f +- %.3f ms on %.2f loud %.2f along %.3f -> %.3f' % (ch, 'tri' if v['kind'] else 'sin', v['rate'], v['mid'] / C.FS * 1e3, abs(v['half']) / C.FS * 1e3, v['on'], v['loud'], v['before'], v['after']))
            if len(keep) == len(V):
                k0 = spiky(y[:, ch]); k1 = spiky(xh)
                for v in V: v['spiky'] = (k0, k1)
                if k0 > 1.0 and k1 < FLAT * k0: notes.append('side %d: %d voices not used, taking them out makes the sound less peaked (%.1f -> %.1f)' % (ch, len(V), k0, k1)); V = []
                break
            V = keep                                                   # some were false: the loudness of the others is measured again without them
        sides.append(V)
    found = any(len(V) > 0 for V in sides); best = max([clear(v) for V in sides for v in V], default=0.0)
    return dict(found=found, sure=float(min(1.0, best / 0.6)) if found else 0.0, voices=sides, n=n, notes=notes)

def apply(y, F, s, speed=1.0, depth=1.0, own=True):
    out = y.copy()
    if not F['found']: return out
    for ch in range(2):
        V = F['voices'][ch]
        if not V: continue
        G = np.array([v['gain'] for v in V]).T; n = len(y)
        if V[0].get('ahead'):                                          # one copy louder than the sound: G holds the sound's level against the copy
            if own:
                if len(y) != F['n']: continue
                H, x, U = backwards(y[:, ch], V, H=G)
                if speed == 1.0 and depth == 1.0: out[:, ch] = y[:, ch] + s * (y[:, ch] - x)
                else: D2 = curves(V, n, speed, depth); out[:, ch] = x + (1.0 + s) * _read(sum(U), np.arange(n, dtype=float) - D2[0])
            else:
                D2 = curves(V, n, speed, depth, F['n']); B = prim.bands(y[:, ch]); z = sum(B[b] / max(G[b, 0], 0.25) for b in range(Gn.NB))
                out[:, ch] = (y[:, ch] + s * _read(z, np.arange(n, dtype=float) - D2[0])) / max(1.0, s / max(V[0]['loud'], 0.25))       # the copy is never put on louder than the sound came in: beyond that the sound itself is turned down
            continue
        if own:
            if len(y) != F['n']: continue
            D = curves(V, n); B = prim.bands(y[:, ch]); xb = [inv._inverse(B[b], D, np.ascontiguousarray(G[b])) for b in range(Gn.NB)]        # the dry sound, range by range
            if speed == 1.0 and depth == 1.0: out[:, ch] = y[:, ch] + s * sum(inv._forward(xb[b], D, np.ascontiguousarray(G[b])) - xb[b] for b in range(Gn.NB))
            else:
                D2 = curves(V, n, speed, depth); out[:, ch] = y[:, ch] + sum((1.0 + s) * (inv._forward(xb[b], D2, np.ascontiguousarray(G[b])) - xb[b]) - (inv._forward(xb[b], D, np.ascontiguousarray(G[b])) - xb[b]) for b in range(Gn.NB))
        else:
            D2 = curves(V, n, speed, depth, F['n']); B = prim.bands(y[:, ch]); out[:, ch] = y[:, ch] + s * sum(inv._forward(B[b], D2, np.ascontiguousarray(G[b])) - B[b] for b in range(Gn.NB))
    return out

def show(F):
    for ch in range(2):
        for v in F['voices'][ch]: print('    side %d %s %.4f Hz  %.3f +- %.3f ms  phase %.3f  on %.2f  %s %.2f (%s) along the curve %.3f -> %.3f, peaked %.1f -> %.1f' % (ch, 'tri' if v['kind'] else 'sin', v['rate'], v['mid'] / C.FS * 1e3, abs(v['half']) / C.FS * 1e3, v['phase'], v['on'], 'LOUDER THAN THE SOUND, the sound against it' if v.get('ahead') else 'loudness', v['loud'], ' '.join('%.2f' % g for g in v['gain']), v['before'], v['after'], *v.get('spiky', (0, 0))), flush=True)
    for l in F.get('notes', []): print('      ' + l, flush=True)

if __name__ == '__main__':
    from lab import load, SETS, score, cases
    names = sys.argv[1:] or [d.rstrip('/').split('/')[-1] for d in cases('chorus')]; rows = []
    for name in names:
        cA = load(SETS + 'chorus/' + name, 'A'); cB = load(SETS + 'chorus/' + name, 'B')
        for c, o, tag in ((cA, cB, 'A'), (cB, cA, 'B')):
            t0 = time.time(); F = find(c['wet']); t1 = time.time(); out = apply(c['wet'], F, -1.0); onto = apply(o['dry'], F, 1.0, own=False); Fd = find(c['dry'])
            r = score.removed(c['dry'], c['wet'], out)[0]; ro = score.removed(o['wet'], o['dry'], onto)[0]; rows.append((r, ro))
            print('%-18s %s found %d sure %.2f | out %6.1f  onto %6.1f | reading %.0f s | on the dry sound: found %d sure %.2f' % (name, tag, F['found'], F['sure'], r, ro, t1 - t0, Fd['found'], Fd['sure']), flush=True); show(F)
            if Fd['found']: print('    ON THE DRY SOUND:'); show(Fd)
    v = np.array(rows); print('chorus: out median %.1f mean %.1f | onto median %.1f mean %.1f (%d sounds)' % (np.median(v[:, 0]), v[:, 0].mean(), np.median(v[:, 1]), v[:, 1].mean(), len(v)))
