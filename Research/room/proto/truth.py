# What really happened in a reverb test case: the click responses give the direct sound and the tail apart, so the dry part and
# the reverb part of the wet sound are known exactly, and with them the true room numbers per band.
import sys, json, os
sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, soundfile as sf, scipy.signal as ss, room
SR = 48000; C = '/tmp/claude-0/v2/corpus/dry/'
def rd(p): return sf.read(p, always_2d=True)[0].astype(np.float64)
def load(d, which='A'):
    """One test case. dry and wet are (n,2) on the same time line; h[i] is the response (n,2) to a click on input channel i."""
    m = json.load(open(d + 'meta.json')); wet = rd(d + which + '_wet.flac') / m['gain' + which]; x = rd(C + m[which] + '.wav')
    dry = np.zeros_like(wet); dry[:len(x)] = x; k = m['click']; h = [rd(d + 'ir_L.flac')[k:] / m['gainIR'], rd(d + 'ir_R.flac')[k:] / m['gainIR']]
    return dict(meta=m, dry=dry, wet=wet, h=h, name=os.path.basename(d.rstrip('/')))
def direct_window(h):
    """Samples that count as the direct sound: 1 ms before to 2.5 ms after the strongest arrival."""
    e = sum((hh ** 2).sum(1) for hh in h); pk = int(np.argmax(e)); return max(0, pk - 48), pk + 120, pk
def parts(c):
    """The wet sound of a case taken apart: (direct part, reverb part), each (n,2); direct + reverb = wet exactly."""
    a, b, pk = direct_window(c['h']); n = len(c['wet'])
    direct = np.stack([ss.fftconvolve(c['dry'][:, 0], c['h'][0][a:b, o])[:n] + ss.fftconvolve(c['dry'][:, 1], c['h'][1][a:b, o])[:n] for o in range(2)], 1)
    if a > 0: direct = np.vstack([np.zeros((a, 2)), direct])[:n]
    return direct, c['wet'] - direct
def room_truth(c):
    """True numbers per band for this case and this sound: tau (s) from the click response's energy decay, kappa = reverb energy
    next to direct energy as it happened on the sound, A = kappa / tau, and the oracle power tracks (frames, bands)."""
    a, b, pk = direct_window(c['h']); hs = c['h'][0] + c['h'][1]; tail = hs.copy(); tail[:b] = 0.0
    E = room.band_power(tail); nT, nB = E.shape; t0 = (b + room.NF) // room.HOP; tau = np.full(nB, np.nan)
    for j in range(nB):
        e = E[t0:, j]
        if e.sum() <= 0: continue
        edc = 10 * np.log10(np.cumsum(e[::-1])[::-1] / e.sum() + 1e-30); i0 = int(np.argmax(edc < -5)); i1 = int(np.argmax(edc < -25))
        if i1 <= i0 + 3: continue
        s = np.polyfit(np.arange(i0, i1) * room.DT, edc[i0:i1], 1)[0]
        if s < -1: tau[j] = -4.3429 / s
    direct, rev = parts(c); S = room.band_power(direct); R = room.band_power(rev); kappa = R.sum(0) / np.maximum(S.sum(0), 1e-30)
    quiet = S.sum(0) < 1e-7 * S.sum(); kappa[quiet] = np.nan
    # the same for a click on both channels at once (what the old truth used), to see how much the sound's stereo width matters
    Sd = room.band_power(hs[a:b]); Rd = room.band_power(tail); kclick = Rd.sum(0) / np.maximum(Sd.sum(0), 1e-30)
    return dict(tau=tau, kappa=kappa, kappa_click=kclick, A=kappa / tau, T60=13.8155 * tau, drr_db=-10 * np.log10(np.maximum(kappa, 1e-12)), S=S, R=R, predelay=(pk) / SR)
if __name__ == '__main__':
    import glob
    for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/reverb/*_0[01]/')):
        c = load(d); t = room_truth(c); mid = (room.FC > 450) & (room.FC < 7000)
        print('%-24s fixed %6.1f dB | T60 mid %.2f s | DRR mid %5.1f dB (click on both channels: %5.1f) | T60 low %.2f high %.2f' % (c['name'], c['meta']['fixed_db'], np.nanmedian(t['T60'][mid]), np.nanmean(t['drr_db'][mid]), np.nanmean(-10 * np.log10(t['kappa_click'][mid])), np.nanmedian(t['T60'][:5]), np.nanmedian(t['T60'][-4:])))
