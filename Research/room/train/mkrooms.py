# A bank of simulated rooms for training: shoebox rooms with real wall materials (absorption per octave band), image sources
# plus ray tracing for the late part, a stereo microphone pair (spaced omnis, ORTF or XY cardioids) and two source positions
# (the left and the right channel of the dry sound). Each room: h[in][n, out], 48 kHz, direct sound of the nearer path at level ~1.
import numpy as np, pyroomacoustics as pra, sys, os, time, json
from pyroomacoustics.directivities import Cardioid, Omnidirectional, DirectionVector
from multiprocessing import Pool
SR = 48000; OUT = '/tmp/claude-0/v2/train/rooms/'; os.makedirs(OUT, exist_ok=True)
MATS = [k for cat in pra.materials_data['absorption'].values() for k in cat.keys()]
HARD = ['hard_surface', 'brickwork', 'rough_concrete', 'unpainted_concrete', 'smooth_brickwork_flush_pointing', 'ceramic_tiles', 'limestone_wall', 'concrete_floor', 'marble_floor', 'plasterboard', 'wooden_lining', 'glass_window', 'double_glazing_30mm', 'wood_1.6cm', 'plywood_thin', 'wood_16mm', 'stage_floor', 'wooden_door', 'linoleum_on_concrete', 'gypsum_board', 'facing_brick', 'ceiling_plasterboard', 'rough_lime_wash']
SOFT = [m for m in MATS if m not in HARD]
def one(i):
    f = OUT + 'room_%04d.npz' % i
    if os.path.exists(f): return i, 0.0, 0.0
    rng = np.random.default_rng(9000 + i); t0 = time.time()
    size = np.exp(rng.uniform(np.log(2.5), np.log(28.0))); lx = size * rng.uniform(0.8, 1.3); ly = size * rng.uniform(0.5, 1.0); lz = float(np.clip(2.3 + 0.25 * size * rng.uniform(0.4, 1.2), 2.3, 14.0))
    soft = rng.uniform(0.0, 0.75) ** 1.5                                     # share of soft surfaces: live rooms are common, dead rooms rarer
    walls = {w: str(rng.choice(SOFT) if rng.random() < soft else rng.choice(HARD)) for w in ('ceiling', 'floor', 'east', 'west', 'north', 'south')}
    room = pra.ShoeBox([lx, ly, lz], fs=SR, materials=pra.make_materials(**walls), max_order=int(rng.integers(8, 18)), air_absorption=True, ray_tracing=True)
    room.set_ray_tracing(receiver_radius=0.5, n_rays=int(3000 + 300 * size), energy_thres=1e-7)
    mx, my = lx * rng.uniform(0.3, 0.7), ly * rng.uniform(0.3, 0.7); mz = rng.uniform(1.2, 1.8); az = rng.uniform(0, 360)        # where the pair sits and which way it looks
    dist = float(np.exp(rng.uniform(np.log(0.25), np.log(max(0.6, 0.45 * size))))); spread = rng.choice([0.0, 0.0, rng.uniform(0.2, 2.0)])   # the two sources: one spot, or apart like a pair of speakers
    a = np.radians(az); fwd = np.array([np.cos(a), np.sin(a)]); side = np.array([-np.sin(a), np.cos(a)])
    for k in range(40):
        c = np.array([mx, my]) + dist * fwd; s1 = c - 0.5 * spread * side; s2 = c + 0.5 * spread * side
        if all(0.3 < p[0] < lx - 0.3 and 0.3 < p[1] < ly - 0.3 for p in (s1, s2)): break
        az += 37; a = np.radians(az); fwd = np.array([np.cos(a), np.sin(a)]); side = np.array([-np.sin(a), np.cos(a)])
        if k % 10 == 9: dist *= 0.7; spread *= 0.7
    else: return i, -1.0, 0.0
    sz = rng.uniform(0.8, 1.8); room.add_source([s1[0], s1[1], sz]); room.add_source([s2[0], s2[1], sz])
    tech = str(rng.choice(['ab', 'ab', 'ortf', 'xy', 'wide']))
    if tech in ('ab', 'wide'):
        d = rng.uniform(0.15, 0.5) if tech == 'ab' else rng.uniform(0.5, 1.5); pl = np.array([mx, my]) - 0.5 * d * side; pr = np.array([mx, my]) + 0.5 * d * side
        room.add_microphone([pl[0], pl[1], mz], directivity=Omnidirectional()); room.add_microphone([pr[0], pr[1], mz], directivity=Omnidirectional())
    else:
        d = 0.17 if tech == 'ortf' else 0.0; ang = 55 if tech == 'ortf' else 45; pl = np.array([mx, my]) - 0.5 * d * side; pr = np.array([mx, my]) + 0.5 * d * side
        room.add_microphone([pl[0], pl[1], mz], directivity=Cardioid(DirectionVector(azimuth=az - ang, colatitude=90, degrees=True))); room.add_microphone([pr[0], pr[1], mz], directivity=Cardioid(DirectionVector(azimuth=az + ang, colatitude=90, degrees=True)))
    room.compute_rir(); n = max(len(room.rir[m][s]) for m in range(2) for s in range(2)); n = min(n, SR * 7); h = np.zeros((2, n, 2))
    for s in range(2):
        for m in range(2): r = room.rir[m][s][:n]; h[s, :len(r), m] = r
    e = (h ** 2).sum((0, 2)); k0 = int(np.argmax(e > 1e-3 * e.max())); k0 = max(0, k0 - 48); h = h[:, k0:]; h /= np.abs(h).max() + 1e-12
    # trim the silent end
    e = (h ** 2).sum((0, 2)); c = np.cumsum(e[::-1])[::-1]; end = int(np.argmax(c < 1e-9 * c[0])) if (c < 1e-9 * c[0]).any() else len(e); h = h[:, :max(end, 2400)]
    np.savez_compressed(f, h=h.astype(np.float32), info=json.dumps(dict(dims=[lx, ly, lz], walls=walls, tech=tech, dist=dist, spread=float(spread))))
    return i, time.time() - t0, h.shape[1] / SR
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    with Pool(4) as p:
        for k, r in enumerate(p.imap_unordered(one, range(a, b))):
            if k % 20 == 0 or r[1] < 0: print(k, r, '%.0f s' % (time.time() - t0), flush=True)
