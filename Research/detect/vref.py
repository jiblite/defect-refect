# The verification step in Python, as Source/Core/Verify.h does it: the blocks, each block's inputs, the trees, the score
# shown. verify_sound() runs everything here (each block through the detectors' Python port and evidence.py); from_dump()
# takes the per-window numbers from the C++ (a line of drverify's output) and so checks everything behind the detectors.
import numpy as np
import evidence as EV
SR = 48000; W = 3 * SR
F12 = ['flanger', 'chorus', 'phaser', 'haas', 'width', 'delay', 'reverb', 'distortion', 'bitcrush', 'filtersweep', 'tremolo', 'autopan']
NEW = ['saturation', 'exciter', 'ringmod', 'filter', 'stutter', 'gatedrhythm']
F27 = F12 + ['eq', 'compressor', 'limiter', 'gate', 'transient', 'saturation', 'exciter', 'ringmod', 'freqshift', 'pitchshift', 'wowflutter', 'filter', 'reverse', 'stutter', 'gatedrhythm']
HAND = ['echo_sc', 'echo_ms', 'side_ratio', 'corr', 'haas_z', 'haas_ms', 'pan_r2', 'pan_span', 'clip_frac', 'log_unique', 'hold', 'crest', 'kurt', 'trem_r2', 'trem_depth']
def blocks(n):
    """(first sample, length) of the blocks a sound is judged in: 6 s each, an end piece shorter than 4.5 s stays with the block before it."""
    B = 6 * SR; join = int(4.5 * SR); full = n // B; rest = n - full * B
    if full == 0: return [(0, n)]
    out = [(b * B, B) for b in range(full)]
    if rest < join: out[-1] = (out[-1][0], B + rest)
    else: out.append((full * B, rest))
    return out
def block_windows(starts, b0, ln):
    """Which of the analysed windows (their first samples) lie wholly in the block; a sound of 3 s or less has its one padded window."""
    return [i for i, s in enumerate(starts) if (s == b0 if ln <= W else (s >= b0 and s + W <= b0 + ln))]
def inputs(fx12, fx27, lg12, hand, p12, lg27, p27, echo, ev):
    """One block's inputs by name (verifier.py feat_names). The arrays have a row per window of the block."""
    x = {}
    for key, cols, a in (('lg12', fx12, lg12), ('lg27', fx27, lg27), ('hd', HAND, hand), ('p12', fx12, p12), ('p27', fx27, p27)):
        a = np.asarray(a, np.float32)
        for st, v in (('mean', a.mean(0)), ('max', a.max(0)), ('min', a.min(0))): x.update({'%s_%s_%s' % (key, c, st): vv for c, vv in zip(cols, v)})
    s12 = {f: float(x['p12_%s_max' % f]) for f in fx12}; s27 = {f: float(x['p27_%s_max' % f]) for f in fx27}; sc = float(echo[0])
    for s in (s12, s27):                                          # the scores as they were shown up to 0.4.1: Inspector.h inspect, Estimate.h mergeNew
        if sc > 60 and s['bitcrush'] < 0.5: s['delay'] = max(s['delay'], min(0.99, 0.6 + sc / 1000.0))
    fin = dict(s12); fin.update({f: s27[f] for f in NEW})
    if any(fin[f] > 0.8 for f in NEW):
        for f in fx12:
            if s27[f] < 0.15: fin[f] = min(fin[f], s27[f])
    x.update({'fin_' + f: v for f, v in fin.items()}); x['wf_sc'] = echo[0]; x['wf_ms'] = echo[1]; x.update(ev)
    return {k: np.float32(v) for k, v in x.items()}
def score_blocks(model, per_block):
    """model: export_verify.read(). per_block: the inputs of each block (None: not judged). -> (the score per effect, every judged block's scores)."""
    names, fams = model; out = {f.name: [] for f in fams}
    for x in per_block:
        if x is None: continue
        v = np.array([x[n] for n in names], np.float32)
        for f in fams: out[f.name].append(f.shown(f.raw(v)))
    return {k: (sorted(v, reverse=True)[(len(v) + 1) // 2 - 1] if v else 0.0) for k, v in out.items()}, out      # the highest score that at least half of the blocks reach
def verify_sound(y, model):
    """Everything in Python: every block is taken as a sound of its own."""
    import appdet
    per = []
    for b0, ln in blocks(len(y)):
        yb = y[b0:b0 + ln]; A = appdet.analyze(yb)
        per.append(None if len(A['starts']) == 0 else inputs(appdet.OLD, F27, A['logit12'], A['hand'], A['p12'], A['logit27'], A['p27'], A['echo'], EV.evidence(yb)))
    return score_blocks(model, per)
def from_dump(d, y, model):
    """From the C++'s per-window numbers and echo checks (a line of drverify's output) and the evidence measured here."""
    per = []; evs = []; num = lambda v: np.array([[np.nan if a is None else a for a in r] for r in v], np.float32)
    for b in d['blocks']:
        if not b['win']: per.append(None); evs.append(None); continue
        ev = EV.evidence(y[b['start']:b['start'] + b['len']]); evs.append(ev); g = lambda k: num([d['win'][r][k] for r in b['win']])
        per.append(inputs(d['fx12'], d['fx27'], g('lg12'), g('hand'), g('p12'), g('lg27'), g('p27'), b['echo'], ev))
    return score_blocks(model, per) + (evs,)
