# (cnntrain2.py for pictures of more than one length: CASES=cases,cases_w,cases_s,cases_x,cases_xc. A batch holds
# pictures of one length; the batches of all lengths are mixed within an epoch. NEG=sb_fx: a folder of sounds WITHOUT a
# reverb, for which every epoch prints how much the model takes for room.)
# Trains the small neural "room or sound" model on the training cases; checks every epoch on the scoreboard's reverb cases.
import sys, glob, json, time, os
sys.path.insert(0, '/tmp/claude-0/v2/train')
import numpy as np, torch, net
torch.set_num_threads(int(os.environ.get('THREADS', '3'))); torch.manual_seed(0); np.random.seed(0)
TAG = os.environ.get('TAG', 'unet'); EPOCHS = int(os.environ.get('EPOCHS', '24')); BS = int(os.environ.get('BS', '8')); LR = float(os.environ.get('LR', '2e-3')); CH = tuple(int(v) for v in os.environ.get('CH', '12,16,24,32,48,48').split(','))
PASS2 = os.environ.get('PASS2', '') == '1'; DROPS = os.environ.get('DROPS', '') == '1'
T = '/tmp/claude-0/v2/train/'; CASES = os.environ.get('CASES', 'cases').split(','); SBX = [v for v in os.environ.get('SBX', '').split(',') if v]; NEG = [v for v in os.environ.get('NEG', '').split(',') if v]
def p2(f): d, n = os.path.split(f); return os.path.join(os.path.dirname(d), 'pass2_' + os.path.basename(d), n)      # .../cases/x.npz -> .../pass2_cases/x.npz
def frames_of(f):
    """How many frames the picture in a case file has (0: no picture in it), read from the file's header alone."""
    import zipfile
    with zipfile.ZipFile(f) as zf:
        if 'LE.npy' not in zf.namelist(): return 0
        with zf.open('LE.npy') as fp:
            v = np.lib.format.read_magic(fp); return (np.lib.format.read_array_header_1_0(fp) if v == (1, 0) else np.lib.format.read_array_header_2_0(fp))[0][0]
def load(files):
    # Kept in half precision (as stored) and as the stored pictures, not as model input: a batch is turned into input
    # (net.inputs, written again for torch in xin) and brought to full precision when it is used. One dict per picture
    # length, filled in place (lists of pictures would take the memory twice). Of the second pass's three extra
    # pictures the last two hold one number per band for the whole sound, and are kept as that.
    T_ = [frames_of(f) for f in files]; by = {}
    for t in sorted(set(T_) - {0}):
        n = sum(1 for v in T_ if v == t); B = 41
        by[t] = dict(LE=np.empty((n, t, B), np.float16), LS=np.empty((n, t, B), np.float16), U64=np.empty((n, t, B), np.float16), U256=np.empty((n, t, B), np.float16), top=np.empty(n, np.float32), w=np.empty(n, np.float32), meta=[],
                     P=np.empty((n, t, B), np.float16) if PASS2 else None, Pb=np.empty((n, 2, B), np.float16) if PASS2 else None)
    for f, t in zip(files, T_):
        if t == 0: continue
        z = np.load(f); G = by[t]; i = len(G['meta']); le, ls = z['LE'].astype(np.float32), z['LS'].astype(np.float32)
        p, base = net.proxy(np.zeros_like(le), le.astype(np.float64), ls.astype(np.float64))
        G['LE'][i] = z['LE']; G['LS'][i] = z['LS']; G['U64'][i] = z['U64']; G['U256'][i] = z['U256']; G['top'][i] = le.max(); G['w'][i] = 1.0 / max(base, 1.0); G['meta'].append(json.loads(str(z['info'])))
        if PASS2:
            X = np.load(p2(f))['X']; assert X.shape == (3, t, 41) and float(np.abs(X[1:].astype(np.float32) - X[1:, :1].astype(np.float32)).max()) == 0.0, f
            G['P'][i] = X[0]; G['Pb'][i] = X[1:, 0]
    return [{k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in G.items()} for t, G in sorted(by.items())]
def xin(G, idx):
    """The model's input for the pictures idx of a group (net.inputs, without the DROPS planes)."""
    le = G['LE'][idx].float(); top = le.amax((1, 2), keepdim=True); e = torch.clamp((le - top) / 20.0, -3.5, 0.0) + 1.75; u1 = torch.clamp(G['U64'][idx].float(), -40.0, 0.0) / 20.0 + 1.0; u2 = torch.clamp(G['U256'][idx].float(), -40.0, 0.0) / 20.0 + 1.0
    pos = torch.linspace(-1.0, 1.0, le.shape[2])[None, None, :].expand_as(le); x = torch.stack([e, u1, u2, pos], 1)
    return torch.cat([x, G['P'][idx].float()[:, None], G['Pb'][idx].float()[:, :, None, :].expand(-1, -1, le.shape[1], -1)], 1) if PASS2 else x
def files_of(c, frac=1.0):
    fs = sorted(glob.glob(T + c + '/*.npz'))
    if frac < 1.0: r = np.random.default_rng(5); fs = [fs[i] for i in sorted(r.permutation(len(fs))[:int(round(frac * len(fs)))])]
    return fs
assert not DROPS; FRAC = {k: float(v) for k, v in (kv.split(':') for kv in os.environ.get('FRAC', '').split(',') if kv)}        # FRAC=cases:0.6,cases_w:0.6 : a fixed random part of a folder
t0 = time.time(); tr = load(sum([files_of(c, FRAC.get(c, 1.0)) for c in CASES], [])); sb = load(sorted(glob.glob(T + 'sb/*.npz'))); sbx = {s: load(sorted(glob.glob(T + s + '/*.npz'))) for s in SBX}; neg = {s: load(sorted(glob.glob(T + s + '/*.npz'))) for s in NEG}
sh = lambda v: [tuple(g['LE'].shape) for g in v]
print('loaded', sh(tr), sh(sb), {s: sh(v) for s, v in sbx.items()}, {s: sh(v) for s, v in neg.items()}, '%.0f s' % (time.time() - t0), flush=True)
_z = np.load(sorted(glob.glob(T + 'sb/*.npz'))[0]); assert PASS2 or np.abs(xin(sb[0], torch.tensor([0]))[0].numpy() - net.inputs(_z['LE'].astype(np.float32), _z['U64'].astype(np.float32), _z['U256'].astype(np.float32))).max() < 1e-5
GLOB = int(os.environ.get('GLOB', '0')); model = net.UNet(cin=(7 if PASS2 else 4) + (2 if DROPS else 0), ch=CH, glob=GLOB); print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
INIT = os.environ.get('INIT'); TOT = float(os.environ.get('TOT', '0')); PCT = float(os.environ.get('PCT', '0.1')); BCE = float(os.environ.get('BCE', '3.0'))
ASYM = float(os.environ.get('ASYM', '1.0'))        # a cell left UNDER the true direct level (too deep) counts ASYM times one left over it
def tloss(g, z, le, ls, top):
    if ASYM == 1.0: return net.train_loss(g, z, le, ls, top, BCE)
    fl = (top - 60.0)[:, None, None]; live = (le > fl).float(); out = torch.maximum(le + g, fl); ref = torch.maximum(ls, fl); d = out - ref; l1 = torch.clamp(torch.where(d < 0, -ASYM * d, d), max=30.0 * ASYM)
    t = torch.clamp((le - ref) / net.GMIN, 0.0, 1.0); bce = torch.nn.functional.binary_cross_entropy_with_logits(z, t, reduction='none') - torch.nn.functional.binary_cross_entropy(t, t, reduction='none')
    return l1 / 10.0 + BCE * bce * live
if INIT:
    st = torch.load('/tmp/claude-0/v2/train/%s.pt' % INIT, map_location='cpu', weights_only=False)['state']; k0 = 'enc.0.0.weight'
    if st[k0].shape[1] < model.state_dict()[k0].shape[1]:        # a second pass starts as the first pass: the extra pictures enter with weight zero
        w = torch.zeros_like(model.state_dict()[k0]); w[:, :st[k0].shape[1]] = st[k0]; st[k0] = w
    miss = model.load_state_dict(st, strict=False); print('starting from', INIT, '| new parts:', sorted(set(k.split('.')[0] for k in miss.missing_keys)), flush=True); assert not miss.unexpected_keys
NB_ = [len(g['LE']) // BS for g in tr]; steps = EPOCHS * sum(NB_)
opt = torch.optim.Adam(model.parameters(), LR); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=PCT, div_factor=20.0)
def evaluate(sets=sb):
    model.eval(); out = []
    with torch.no_grad():
        for s in sets:
            for i in range(0, len(s['LE']), 16):
                g = model(xin(s, torch.arange(i, min(i + 16, len(s['LE']))))).numpy()
                for k in range(len(g)):
                    p, base = net.proxy(g[k].astype(np.float64), s['LE'][i + k].numpy().astype(np.float64), s['LS'][i + k].numpy().astype(np.float64)); m = s['meta'][i + k]; out.append((m['split'], m['drr'], p))
    model.train(); r = {}
    for sp in ('dev', 'test'):
        v = np.array([o[2] for o in out if o[0] == sp]); d = np.array([o[1] for o in out if o[0] == sp]); r[sp] = (np.median(v), np.mean(v), 100 * np.mean(v < 0), np.median(v[d >= 8]), np.median(v[d < 0]))
    return r, out
def taken(sets):
    """Sounds without a reverb: what the model takes for room. Per sound, over the cells within 50 dB of the loudest: the
    share it turns down by more than 12 dB, and the share of the energy it takes away (dB under the whole)."""
    model.eval(); deep, share = [], []
    with torch.no_grad():
        for s in sets:
            for i in range(0, len(s['LE']), 16):
                g = model(xin(s, torch.arange(i, min(i + 16, len(s['LE']))))).numpy().astype(np.float64)
                for k in range(len(g)):
                    le = s['LE'][i + k].numpy().astype(np.float64); live = le > le.max() - 50.0; E = 10 ** ((le - le.max()) / 10.0)
                    deep.append(np.mean(g[k][live] < -12.0)); share.append(10 * np.log10(max((E * (1 - 10 ** (g[k] / 10.0)))[live].sum(), 1e-30) / E[live].sum()))
    model.train(); deep = np.array(deep); share = np.array(share)
    return 'deep cells: mean %.3f, over 0.02 on %.0f%%, over 0.1 on %.0f%% | taken away (dB): median %.1f, 90th %.1f' % (deep.mean(), 100 * np.mean(deep > 0.02), 100 * np.mean(deep > 0.1), np.median(share), np.percentile(share, 90))
def parts_line(sets=None):
    """Scoreboard reverbs (849 frames): the sound itself (first 562 frames) and the ring-out, each against its own untouched distance."""
    sets = sets or sb; model.eval(); rows = []
    with torch.no_grad():
        for s in sets:
            if s['LE'].shape[1] < 800: continue
            for i in range(0, len(s['LE']), 16):
                g = model(xin(s, torch.arange(i, min(i + 16, len(s['LE']))))).numpy().astype(np.float64)
                for k in range(len(g)):
                    LE = s['LE'][i + k].numpy().astype(np.float64); LS = s['LS'][i + k].numpy().astype(np.float64); fl = LE.max() - 60.0; ref = np.maximum(LS, fl); a = np.minimum(np.abs(np.maximum(LE + g[k], fl) - ref), 30.0); b = np.minimum(np.abs(np.maximum(LE, fl) - ref), 30.0); m = s['meta'][i + k]
                    rows.append((m['split'], m['drr'], 100 * (1 - a[:562].sum() / max(b[:562].sum(), 1e-9)), 100 * (1 - a[562:].sum() / max(b[562:].sum(), 1e-9)), 100 * (b[:562] - a[:562])[(LE[:562] + g[k][:562] < ref[:562] - 3.0)].sum() / max(b[:562].sum(), 1e-9)))
    model.train(); out = []
    for sp in ('dev', 'test'):
        for lab, sel in (('all', lambda d: True), ('light', lambda d: d >= 5)):
            v = np.array([r[2:] for r in rows if r[0] == sp and sel(r[1])]); out.append('%s %s: sound itself %.1f / %.1f / %.0f%% (too deep costs %.1f), ring-out %.1f' % (sp, lab, np.median(v[:, 0]), v[:, 0].mean(), 100 * (v[:, 0] < 0).mean(), -v[:, 2].mean(), np.median(v[:, 1])))
    return ' | '.join(out)
for s_, v in neg.items(): print('before | %s (%d sounds without a reverb) %s' % (s_, sum(len(g['LE']) for g in v), taken(v)), flush=True)
print('before | ' + parts_line(), flush=True)
r, out = evaluate(); print('before | dev: median %.1f mean %.1f below zero %.0f%% (light %.1f, wet %.1f) | unseen: median %.1f mean %.1f below zero %.0f%% (light %.1f, wet %.1f)' % (r['dev'] + r['test']) + ''.join(' | %s dev: %.1f %.1f %.0f%%, unseen: %.1f %.1f %.0f%%' % ((s_,) + evaluate(v)[0]['dev'][:3] + evaluate(v)[0]['test'][:3]) for s_, v in sbx.items()), flush=True)
best = -1e9; step = 0; START = int(os.environ.get('START', '0'))        # START=3: go on after three finished epochs (INIT = the <tag>_last of that run); the step size picks up where it was
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for _ in range(START * sum(NB_)): sched.step(); step += 1
for ep in range(START, EPOCHS):
    order = [(gi, torch.randperm(len(g['LE']))) for gi, g in enumerate(tr)]; plan = sum([[(gi, k) for k in range(NB_[gi])] for gi in range(len(tr))], []); plan = [plan[i] for i in torch.randperm(len(plan)).tolist()]; tot = 0.0; te = time.time()
    for gi, k in plan:
        G = tr[gi]; idx = order[gi][1][k * BS:(k + 1) * BS]; le, ls = G['LE'][idx].float(), G['LS'][idx].float(); g, z = model(xin(G, idx)); l = (tloss(g, z, le, ls, G['top'][idx]).mean((1, 2)) * G['w'][idx]).mean()
        if TOT > 0: l = l + TOT * net.total_loss(g, le, ls, G['top'][idx]).mean()
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step(); tot += float(l.detach()); step += 1
        if step == START * sum(NB_) + 20: print('  20 steps in %.0f s' % (time.time() - te), flush=True)
    r, out = evaluate(); sc = r['dev'][0] + r['dev'][1] + r['test'][0] * 0     # model choice uses the dev reverbs only
    more = ''
    for s_, v in sbx.items(): rx, _ = evaluate(v); sc += 0.5 * (rx['dev'][0] + rx['dev'][1]); more += ' | %s dev: median %.1f mean %.1f below zero %.0f%%, unseen: %.1f %.1f %.0f%%' % ((s_,) + rx['dev'][:3] + rx['test'][:3])
    print('epoch %2d  loss %.4f  %.0f s | dev: median %.1f mean %.1f below zero %.0f%% (light %.1f, wet %.1f) | unseen: median %.1f mean %.1f below zero %.0f%% (light %.1f, wet %.1f)' % ((ep, tot / len(plan), time.time() - te) + r['dev'] + r['test']) + more, flush=True)
    for s_, v in neg.items(): print('         | %s %s' % (s_, taken(v)), flush=True)
    print('         | ' + parts_line(), flush=True)
    torch.save(dict(cfg=model.cfg, state=model.state_dict(), kind='unet'), '/tmp/claude-0/v2/train/%s_last.pt' % TAG); torch.save(dict(cfg=model.cfg, state=model.state_dict(), kind='unet'), '/tmp/claude-0/v2/train/%s_e%d.pt' % (TAG, ep))
    if sc > best: best = sc; torch.save(dict(cfg=model.cfg, state=model.state_dict(), kind='unet'), '/tmp/claude-0/v2/train/%s.pt' % TAG); json.dump(out, open('/tmp/claude-0/v2/train/%s.sb.json' % TAG, 'w'))
print('done, %.0f s' % (time.time() - t0))
