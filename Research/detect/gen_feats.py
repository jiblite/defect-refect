# Features for the verification step, for training (dev clips, extra clean sounds, the app's own effects and the dev-split
# outside effects) and for the held-out test (test clips, test-split outside effects). The sounds are made, measured and
# thrown away; one pickle of rows per unit under data/feats/.      python3 gen_feats.py train|pos2|test PART NPARTS
#   per sound: everything the app's detection computes (appdet.analyze: per-window pictures' logits, hand numbers, deciders'
#   scores, the app's final scores; the CNNs' embeddings averaged over the windows) and the whole-sound evidence (evidence.py)
import sys, os, json, glob, hashlib, pickle, time, subprocess, tempfile, shutil
import paths
import numpy as np, soundfile as sf, scipy.signal as ss
import appdet, evidence, own_fx, extra_clean as X, zoo_fx as Z
from cpuslot import slot
SR = 48000; paths.make_work(); D = paths.WORK; CORPUS = paths.CORPUS; SETS = paths.SETS; OUT = D + 'data/feats/'
meta = json.load(open(CORPUS + 'meta.json')); FAMS = own_fx.FAMILIES
DEV = sorted(k for k in meta if meta[k]['split'] == 'dev'); TEST = sorted(k for k in meta if meta[k]['split'] == 'test')
ZDEV = { f: sorted(n for n, z in Z.ZOO.items() if z[0] == f and z[1] == 'dev') for f in FAMS }; ZTEST = { f: sorted(n for n, z in Z.ZOO.items() if z[0] == f and z[1] == 'test') for f in FAMS }
def dry(name): return sf.read(CORPUS + 'dry/' + name + '.wav', always_2d=True)[0].astype(np.float64)
def h32(s): return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 ** 32
# two excerpts of one long recording belong together: one group (named after the first of them), never split between folds
_up = {}
for k in sorted(meta):
    if meta[k]['kind'] == 'real-recording': _up.setdefault(meta[k]['note'].split(' at ')[0], []).append(k)
GROUP = { k: v[0] for v in _up.values() for k in v }
SIBLING = { k for v in _up.values() if len({ meta[c]['split'] for c in v }) > 1 for k in v }          # recordings with excerpts on both sides of the dev/test split
def group_of(name): return GROUP.get(name, name)
def fold_of(name): return h32('fold/' + group_of(name)) % 5

def sources():
    out = []; xi = 0
    for k in DEV:
        out.append(('dev', k))
        for _ in range(2):
            if xi < X.COUNT: out.append(('x', xi)); xi += 1
    return out
SRC = sources(); SRCNAME = [s[1] if s[0] == 'dev' else X.name_of(s[1]) for s in SRC]
def src_audio(s):
    if s[0] == 'dev': return s[1], meta[s[1]]['kind'], dry(s[1])
    return X.make(s[1])

def feats(y):
    A = appdet.analyze(y, emb=True); E = evidence.evidence(y); W = len(A['starts'])
    if not os.path.exists(D + 'data/ev_names.json'): json.dump(evidence.NAMES, open(D + 'data/ev_names.json', 'w'))       # the order of 'ev', for verifier.py
    r = dict(n=len(y), nwin=W, echo=np.array(A['echo'], np.float32), hand=A['hand'].astype(np.float32), logit12=A['logit12'].astype(np.float32), logit27=A['logit27'].astype(np.float32),
             p12=A['p12'].astype(np.float32), p27=A['p27'].astype(np.float32), final=np.array([A['final'][f] for f in appdet.OLD + appdet.NEW], np.float32), ev=np.array([E[k] for k in evidence.NAMES], np.float32))
    if W: r.update(emb12m=A['emb12'].mean(0).astype(np.float16), emb12x=A['emb12'].max(0).astype(np.float16), emb27m=A['emb27'].mean(0).astype(np.float16), emb27x=A['emb27'].max(0).astype(np.float16))
    return r
def db(a, b): return float(10 * np.log10((a ** 2).sum() / max((b ** 2).sum(), 1e-30) + 1e-30))
def change_db(x, w):
    n = min(len(x), len(w)); g = (w[:n] * x[:n]).sum() / ((x[:n] ** 2).sum() + 1e-30); return db(w[:n] - g * x[:n], x[:n] * g) if abs(g) > 1e-3 else 99.0
def fade(y, k=480): y = y.copy(); k = min(k, len(y) // 2); r = np.linspace(0, 1, k); y[:k] *= r[:, None]; y[-k:] *= r[::-1, None]; return y
def is_stereo(x): return ((x[:, 0] - x[:, 1]) ** 2).mean() > 1e-6 * (x ** 2).mean()
def finalize(y, r):
    """Training sounds come at any level and in any of the usual sample formats, whatever was done to them."""
    pk = np.abs(y).max() + 1e-12; y = y * (float(np.exp(r.uniform(np.log(0.1), np.log(0.95)))) / pk); f = r.choice(['f32', 'f32', 'f32', 'i16', 'i16', 'i24'])
    if f == 'i16': y = np.round(y * 32767 + (r.random(y.shape) - r.random(y.shape) if r.random() < 0.5 else 0)) / 32767
    if f == 'i24': y = np.round(y * 8388607) / 8388607
    return y.astype(np.float32).astype(np.float64), f
def codec(x, r):
    d = tempfile.mkdtemp(dir=D + 'tmp/zoo'); a = d + '/a.wav'; sf.write(a, x.astype(np.float32), SR, subtype='FLOAT')
    try:
        kind = r.choice(['mp3', 'mp3', 'ogg', 'aac']); b = d + '/b.' + {'mp3': 'mp3', 'ogg': 'ogg', 'aac': 'm4a'}[kind]
        enc = {'mp3': ['-c:a', 'libmp3lame', '-b:a', str(r.choice(['96k', '128k', '192k', '256k']))], 'ogg': ['-c:a', 'libvorbis', '-q:a', str(int(r.integers(2, 7)))], 'aac': ['-c:a', 'aac', '-b:a', str(r.choice(['96k', '128k', '192k']))]}[kind]
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a] + enc + [b], check=True, capture_output=True); subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', b, '-ar', str(SR), '-c:a', 'pcm_f32le', d + '/c.wav'], check=True, capture_output=True)
        y = sf.read(d + '/c.wav', always_2d=True)[0].astype(np.float64); return (y if y.shape[1] == 2 else np.repeat(y[:, :1], 2, 1)), kind
    finally: shutil.rmtree(d, ignore_errors=True)
def speed(x, r):
    q = 2 ** (float(r.choice([-1, 1]) * r.uniform(1.5, 5)) / 12); y = ss.resample_poly(x, 1000, int(round(1000 * q)), axis=0)
    if len(y) > 6 * SR: s = int(r.integers(0, len(y) - 6 * SR + 1)); y = y[s:s + 6 * SR]
    return fade(y)
def partner(name, r):
    f = fold_of(name); c = [i for i, n in enumerate(SRCNAME) if fold_of(n) == f and group_of(n) != group_of(name)]
    for _ in range(5):
        nm, _, y = src_audio(SRC[int(r.choice(c))])
        if y is not None: return nm, y
    return None, None
def mixed(x, name, r):
    nm, o = partner(name, r)
    if o is None: return None, None
    g = 10 ** (r.uniform(-9, 3) / 20) * np.sqrt((x ** 2).mean() / ((o ** 2).mean() + 1e-20)); return x + g * o, nm

CASES = {}
def cases(fam, split):
    if (fam, split) not in CASES: CASES[(fam, split)] = [d for d in sorted(glob.glob(SETS + fam + '/*/')) if json.load(open(d + 'meta.json'))['split'] == split]
    return CASES[(fam, split)]
def through_ir(x, d, tail=3.0):
    """A stored case's click answers (reverb or tone, a fixed linear effect) put on another sound."""
    m = json.load(open(d + 'meta.json')); k = m['click']; n = len(x) + int(tail * SR); L = int((tail + 0.5) * SR)
    h = [sf.read(d + 'ir_%s.flac' % c, always_2d=True)[0][k:k + L].astype(np.float64) / m['gainIR'] for c in 'LR']
    return np.stack([ss.fftconvolve(x[:, 0], h[0][:, o])[:n] + ss.fftconvolve(x[:, 1], h[1][:, o])[:n] for o in range(2)], 1)
def case_sound(d, which): m = json.load(open(d + 'meta.json')); return sf.read(d + which + '_wet.flac', always_2d=True)[0].astype(np.float64) / m['gain' + which], m

def one_effect(x, fam, r, split='dev', own=None):
    """One audible effect of a family on x: (wet, tag, params) or None. own: True the app's own, False an outside one of the split, None either."""
    names = [n for n in (ZDEV if split == 'dev' else ZTEST)[fam] if Z.ZOO[n][4] is None or Z.ZOO[n][4](x)]
    if own is None: own = r.random() < 0.5
    if not own and not names: own = True if split == 'dev' else None
    if own is None: return None
    for _ in range(4):
        try:
            if own: p = own_fx.sample(fam, r); w = own_fx.apply(fam, x, p, 3.0 if fam == 'delay' else 0.0); tag = 'own'
            else: tag = str(r.choice(names)); p = Z.ZOO[tag][2](r); w = Z.apply(tag, x, p, Z.TAIL.get(fam, 0.0))
        except Exception as e: continue
        if not np.isfinite(w).all() or np.abs(w).max() < 1e-3: continue
        if change_db(x, w) > -30: return w, tag, p
    return None

def rows_for_source(k):
    s = SRC[k]; name, kind, x = src_audio(s)
    if x is None: return []
    r = np.random.default_rng(h32('train/' + name)); out = []; base = dict(set='train', src=name, group=group_of(name), fold=fold_of(name), kind=kind)
    def add(y, variant, labels=(), impl=(), params=(), **kw):
        y, fmt = finalize(y, r)
        with slot(): f = feats(y)
        row = dict(base); row.update(id='%s/%s' % (name, variant), variant=variant, labels=list(labels), impl=list(impl), params=json.dumps(params, default=str), fmt=fmt); row.update(kw); row.update(f); out.append(row)
    add(x, 'raw'); st = is_stereo(x)
    opts = ['seg', 'speed', 'speed', 'rev', 'mix', 'mix', 'mix', 'pad', 'cat', 'codec', 'bal'] + (['mono', 'mono', 'narrow'] if st else []); pick = []
    while len(pick) < 3:
        v = str(r.choice(opts))
        if v not in pick: pick.append(v)
    for v in pick:
        try:
            with slot():
                extra = {}
                if v == 'mono': m = x.mean(1); y = np.stack([m, m], 1)
                elif v == 'narrow': m = x.mean(1); sd = 0.5 * (x[:, 0] - x[:, 1]) * r.uniform(0.2, 0.7); y = np.stack([m + sd, m - sd], 1)
                elif v == 'bal': g = r.uniform(0.4, 0.9); y = x * ([1.0, g] if r.random() < 0.5 else [g, 1.0])
                elif v == 'seg': L = int(r.uniform(3.2, 5.0) * SR); s0 = int(r.integers(0, len(x) - L)); y = fade(x[s0:s0 + L])
                elif v == 'speed': y = speed(x, r)
                elif v == 'rev': y = x[::-1].copy()
                elif v == 'mix': y, nm = mixed(x, name, r); extra['with'] = nm
                elif v == 'pad': a = int(float(r.choice([0, 0, r.uniform(0.3, 1.5)])) * SR); b = int(r.uniform(1.0, 3.0) * SR); y = np.vstack([np.zeros((a, 2)), x, np.zeros((b, 2))])
                elif v == 'cat': nm, o = partner(name, r); y = np.vstack([fade(x[:3 * SR]), fade(o[3 * SR:])]) if o is not None else None; extra['with'] = nm
                elif v == 'codec': y, c = codec(x, r); extra['codec'] = c
            if y is None: continue
            add(y, v, **extra)
        except Exception as e: print('variant failed', name, v, repr(e)[:200], flush=True)
    for fam in ('reverb', 'tone'):
        try:
            d = str(r.choice(cases(fam, 'dev')))
            with slot(): y = through_ir(x, d, 3.0 if fam == 'reverb' else 0.5)
            add(y, fam + '_ir', case=os.path.basename(d[:-1]))
        except Exception as e: print('ir failed', name, fam, repr(e)[:200], flush=True)
    for j in range(8):
        fam = FAMS[(k * 8 + j) % len(FAMS)]
        try:
            with slot():
                b = x; bt = 'raw'
                u = r.random()
                if u < 0.15: b = speed(x, r); bt = 'speed'
                elif u < 0.30:
                    b2, nm = mixed(x, name, r)
                    if b2 is not None: b = b2; bt = 'mix'
                e = one_effect(b, fam, r, 'dev', own=((k + j) % 2 == 0))
                if e is None: print('no audible', name, fam, flush=True); continue
                w, tag, p = e; labels = [fam]; impl = [tag]; params = [p]; u = r.random()
                if u < 0.10:                                   # reverb or a tone change on top (not one of the 17: the labels stay)
                    f2 = str(r.choice(['reverb', 'tone'])); d = str(r.choice(cases(f2, 'dev'))); w = through_ir(w, d, 3.0 if f2 == 'reverb' else 0.5); impl.append(os.path.basename(d[:-1]))
                elif u < 0.25:                                 # a second effect of another family behind the first
                    f2 = str(r.choice([f for f in FAMS if f != fam])); e2 = one_effect(w, f2, r, 'dev')
                    if e2 is not None: w, t2, p2 = e2; labels.append(f2); impl.append(t2); params.append(p2)
            add(w, 'fx%d_%s' % (j, '+'.join(labels)), labels, impl, params, base_variant=bt)
        except Exception as e: print('effect failed', name, fam, repr(e)[:200], flush=True)
    return out

WEAK = ['flanger', 'phaser', 'chorus', 'filtersweep', 'tremolo', 'filter', 'saturation', 'exciter']
def rows_pos2(k):
    """A second helping of positives for the families that are hardest to tell from clean sounds: five more per source."""
    s = SRC[k]; name, kind, x = src_audio(s)
    if x is None: return []
    r = np.random.default_rng(h32('pos2/' + name)); out = []; base = dict(set='train', src=name, group=group_of(name), fold=fold_of(name), kind=kind)
    for j in range(5):
        fam = WEAK[(k * 5 + j) % len(WEAK)]
        try:
            with slot():
                b = x; bt = 'raw'; u = r.random()
                if u < 0.2: b = speed(x, r); bt = 'speed'
                elif u < 0.4:
                    b2, nm = mixed(x, name, r)
                    if b2 is not None: b = b2; bt = 'mix'
                e = one_effect(b, fam, r, 'dev', own=((k + j) % 2 == 1))
                if e is None: print('no audible', name, fam, flush=True); continue
                w, tag, p = e; impl = [tag]
                if r.random() < 0.12: f2 = str(r.choice(['reverb', 'tone'])); d = str(r.choice(cases(f2, 'dev'))); w = through_ir(w, d, 3.0 if f2 == 'reverb' else 0.5); impl.append(os.path.basename(d[:-1]))
                y, fmt = finalize(w, r); f = feats(y)
            row = dict(base); row.update(id='%s/p2_%d_%s' % (name, j, fam), variant='p2_%d_%s' % (j, fam), labels=[fam], impl=impl, params=json.dumps([p], default=str), fmt=fmt, base_variant=bt); row.update(f); out.append(row)
        except Exception as e: print('effect failed', name, fam, repr(e)[:200], flush=True)
    return out
def chunks(v, n): return [v[i:i + n] for i in range(0, len(v), n)]
def stored_rows(items, setname):
    out = []
    for fam, d, which in items:
        y, m = case_sound(d, which); clip = m[which]; pos = fam not in ('reverb', 'tone')
        with slot(): f = feats(y)
        row = dict(set=setname, src=clip, group=group_of(clip), fold=fold_of(clip), kind=meta[clip]['kind'], id='%s/%s/%s' % (fam, os.path.basename(d[:-1]), which), variant='case_' + fam, labels=[fam] if pos else [],
                   impl=[m['effect']], params=json.dumps(m['params'], default=str), split=m['split'], sibling=clip in SIBLING, stored=True); row.update(f); out.append(row)
    return out
def stored_items(fams, split): return [(fam, d, w) for fam in fams for d in cases(fam, split) for w in 'AB']

def test_dry_rows(names):
    out = []
    for c in names:
        x = dry(c); r = np.random.default_rng(h32('testdry/' + c))
        for v in ('dry', 'drypad', 'dry16', 'drymp3'):
            try:
                with slot():
                    y = x if v == 'dry' else np.vstack([x, np.zeros((3 * SR, 2))]) if v == 'drypad' else (np.round(x * 32767) / 32767).astype(np.float32).astype(np.float64) if v == 'dry16' else codec(x, np.random.default_rng(1))[0]
                    f = feats(y)
                row = dict(set='test', src=c, group=group_of(c), fold=-1, kind=meta[c]['kind'], id='%s/%s' % (v, c), variant=v, labels=[], impl=[], params='[]', split='test', sibling=c in SIBLING); row.update(f); out.append(row)
            except Exception as e: print('test dry failed', c, v, repr(e)[:200], flush=True)
    return out
def test_mix_rows():
    out = []; r = np.random.default_rng(77); order = r.permutation(len(TEST))
    for i in range(0, len(order) - 1, 2):
        a, b = TEST[order[i]], TEST[order[i + 1]]; xa, xb = dry(a), dry(b); g = 10 ** (r.uniform(-6, 6) / 20) * np.sqrt((xa ** 2).mean() / ((xb ** 2).mean() + 1e-20)); y = xa + g * xb; y = y * (0.5 / np.abs(y).max())
        with slot(): f = feats(y)
        row = dict(set='test', src=a, group=group_of(a), fold=-1, kind='mix', id='drymix/%s+%s' % (a, b), variant='drymix', labels=[], impl=[], params='[]', split='test', sibling=(a in SIBLING or b in SIBLING)); row.update(f); out.append(row)
    return out
def test_ext_rows(fam, part, nper=36, nparts=4):
    """Held-out positives beyond the stored scoreboard cases: test-split outside effects on test-split clips, fresh settings."""
    out = []; names = ZTEST[fam]
    for i in range(part, nper, nparts):
        name = names[i % len(names)]; r = np.random.default_rng(h32('ext/%s/%d' % (name, i))); clipf = Z.ZOO[name][4]
        try:
            with slot():
                ok = None
                for _ in range(8):
                    c = str(r.choice(TEST)); x = dry(c)
                    if clipf is not None and not clipf(x): continue
                    p = Z.ZOO[name][2](r); w = Z.apply(name, x, p, Z.TAIL.get(fam, 0.0))
                    if np.isfinite(w).all() and np.abs(w).max() > 1e-3 and change_db(x, w) > -30: ok = (c, p, w); break
                if ok is None: print('no audible', fam, name, i, flush=True); continue
                c, p, w = ok; f = feats(w)
            row = dict(set='test', src=c, group=group_of(c), fold=-1, kind=meta[c]['kind'], id='ext/%s/%s_%02d' % (fam, name, i), variant='ext_' + fam, labels=[fam], impl=[name], params=json.dumps(p, default=str), split='test', sibling=c in SIBLING, stored=False); row.update(f); out.append(row)
        except Exception as e: print('ext failed', fam, name, i, repr(e)[:200], flush=True)
    return out

def units(which):
    u = []
    if which == 'train':
        u += [('src_%03d' % k, (lambda k=k: rows_for_source(k))) for k in range(len(SRC))]
        u += [('casesdev_%02d' % i, (lambda c=c: stored_rows(c, 'train'))) for i, c in enumerate(chunks(stored_items(['reverb', 'tone'], 'dev'), 16))]
    elif which == 'pos2': u += [('pos2_%03d' % k, (lambda k=k: rows_pos2(k))) for k in range(0, len(SRC), 2)]
    else:
        u += [('t_dry_%02d' % i, (lambda c=c: test_dry_rows(c))) for i, c in enumerate(chunks(TEST, 6))]
        u += [('t_cases_%02d' % i, (lambda c=c: stored_rows(c, 'test'))) for i, c in enumerate(chunks(stored_items(['reverb', 'tone'], 'test'), 16))]
        u += [('t_pos_%s_%s' % (f, sp), (lambda f=f, sp=sp: stored_rows(stored_items([f], sp), 'test' if sp == 'test' else 'posdev'))) for f in FAMS for sp in ('test', 'dev')]
        u += [('t_ext_%s_%d' % (f, q), (lambda f=f, q=q: test_ext_rows(f, q))) for q in range(4) for f in FAMS]
        u += [('t_mix', test_mix_rows)]
    return u
if __name__ == '__main__':
    which, part, nparts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(OUT, exist_ok=True); U = units(which); t0 = time.time(); nd = 0
    only = sys.argv[4:]
    for i, (name, fn) in enumerate(U):
        if i % nparts != part or (only and not any(name.startswith(o) for o in only)): continue
        path = OUT + name + '.pkl'
        if os.path.exists(path): continue
        t1 = time.time(); rows = fn(); pickle.dump(rows, open(path + '.tmp', 'wb')); os.replace(path + '.tmp', path); nd += len(rows)
        print('%s: %d rows, %.0f s (%d rows in %.0f s so far)' % (name, len(rows), time.time() - t1, nd, time.time() - t0), flush=True)
    print('finished', flush=True)
