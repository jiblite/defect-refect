# A larger set of stereo cases made with the outside effects (zoo_fx.py): n draws per effect, any clip of the corpus.
#   python3 st7.py make n          -> gen/<effect>_<i>.flac, gen/meta.json
#   python3 st7.py run i n         -> st7_<i>.jsonl (the engine's numbers, removal score against the dry clip)
import sys, os, json, hashlib, time
sys.path.insert(0, '/tmp/claude-0/v2/newfx'); sys.path.insert(0, '/home/claude/defect-refect/Research/detect')
from st_common import *
G = '/tmp/claude-0/v2/newfx/gen/'; os.makedirs(G, exist_ok=True)
NAMES = ['haas_sox', 'haas_calf_stereotools', 'haas_ffmpeg', 'haas_mda', 'wid_caps', 'wid_calf_haasenh', 'wid_mda_comb', 'wid_calf_base']
if sys.argv[1] == 'make':
    import zoo_fx as Z
    cm = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); clips = sorted(cm); meta = {}
    for name in NAMES:
        fam, split, sampler, fn, clipf = Z.ZOO[name]
        for i in range(int(sys.argv[2])):
            rng = np.random.default_rng(int(hashlib.md5((name + 'gen/%d' % i).encode()).hexdigest(), 16) % 2 ** 32)
            for tries in range(8):
                p = sampler(rng); a = str(rng.choice(clips)); x = truth.rd('/tmp/claude-0/v2/corpus/dry/' + a + '.wav')
                if clipf is not None and not clipf(x): continue
                try: w = Z.apply(name, x, p, 0.0)
                except Exception as e: print(name, i, 'failed', str(e)[:100]); continue
                if np.isfinite(w).all() and np.abs(w).max() > 1e-3: break
            else: print(name, i, 'no case'); continue
            g = 1.0 if np.abs(w).max() < 0.98 else 0.98 / np.abs(w).max(); sf.write(G + '%s_%02d.flac' % (name, i), w * g, SR, format='FLAC', subtype='PCM_24')
            meta['%s_%02d' % (name, i)] = dict(effect=name, family=fam, clip=a, params=p, gain=g); print(name, i, a if not a.startswith('real_') else 'real_XX', flush=True)
    json.dump(meta, open(G + 'meta.json', 'w'), indent=1)
else:
    import sides as E
    from st5 import numbers
    meta = json.load(open(G + 'meta.json')); i0, n0 = int(sys.argv[2]), int(sys.argv[3]); out = open('st7_%d.jsonl' % i0, 'w')
    for nm in sorted(meta)[i0::n0]:
        m = meta[nm]; wet = truth.rd(G + nm + '.flac') / m['gain']; dry = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['clip'] + '.wav'); n = min(len(wet), len(dry)); wet, dry = wet[:n], dry[:n]
        t0 = time.time(); f = E.find(wet); row = dict(name=nm, effect=m['effect'], clip=m['clip'], dry_side=db(*ms(dry)[::-1]) if (ms(dry)[0] ** 2).sum() > 0 else 99.0, secs=time.time() - t0); r = numbers(f); r['found'] = r.pop('kind'); row.update(r)
        if f['mode'] is not None:
            o = E.own(wet, f, 0.0); row['out'] = score.removed(dry, wet, o)[0]; row['out4'] = removed4(dry, wet, o)[0]
        mono = np.stack([dry.mean(1)] * 2, 1); row['best'] = score.removed(dry, wet, mono)[0]; row['best4'] = removed4(dry, wet, mono)[0]        # what the dry sound folded to mono would score
        out.write(json.dumps(row) + '\n'); out.flush()
