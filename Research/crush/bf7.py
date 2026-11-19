# The crush engine on the fresh cases (fc7.py make bitcrush n): what is read against what was set, and the crusher
# put onto clip B against B through the same effect.   python3 bf7.py run [py|app] ; python3 bf7.py show
import sys, os, json, glob, time, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import crush as E
G = '/tmp/claude-0/v2/newfx/gen_bitcrush/'; HERE = '/tmp/claude-0/v2/newfx/'
def want(m):
    """(step, hold) as set, where the tool's rule is known."""
    p = m['params']; e = m['effect']
    if e == 'plain': return p['step'], p['P']
    if e == 'bit_pedalboard': return 2.0 ** (-1.0 - p['bits']), float(p['down'])
    if e == 'bit_swh': return 0.5 * 2.0 ** (0.999 - p['bits']), float(p['down'])
    if e == 'bit_calf': return 1.0 / (2.0 ** p['bits'] - 1.0), float(p['samples'])
    return None, None
if sys.argv[1] == 'run':
    how = sys.argv[2] if len(sys.argv) > 2 else 'py'; meta = json.load(open(G + 'meta.json')); out = open(HERE + 'bf7_%s.jsonl' % how, 'w')
    for nm in sorted(meta):
        m = meta[nm]; ya = truth.rd(G + nm + '_A.flac') / m['gain']; yb = truth.rd(G + nm + '_B.flac') / m['gain']; xa = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['A'] + '.wav'); xb = truth.rd('/tmp/claude-0/v2/corpus/dry/' + m['B'] + '.wav'); t0 = time.time()
        if how == 'py':
            f = E.find(ya); found = f['found']; o = E.onto(xb, f) if found else None; J = dict(step=f.get('step'), offset=f.get('offset'), P=f.get('P'), c=f.get('c'), bits=f.get('bits'), kind=f.get('kind'), mix=f.get('mix'), soft=f.get('soft'))
            J['grids'] = [None if g is None else dict(step=g['step'], share=g['share'], levels=g['levels']) for g in f.get('grids', [])]; J['hold'] = {k: v for k, v in f.get('hold', {}).items() if k in ('P', 'c', 'share', 'changes')}
            back = E.undo(ya, f) if found and hasattr(E, 'undo') else None
        else:
            tmp = tempfile.mkdtemp(dir='/dev/shm'); pa, pb, pre = tmp + '/a.wav', tmp + '/b.wav', tmp + '/o'; sf.write(pa, ya, SR, subtype='FLOAT'); sf.write(pb, xb, SR, subtype='FLOAT')
            r = subprocess.run(['/tmp/claude-0/drcrush', pa, pb, pre], capture_output=True, text=True); J = json.loads(r.stdout.strip().splitlines()[-1]); found = J['found']
            o = np.fromfile(pre + '.onto.f64').reshape(2, -1).T if found else None; back = np.fromfile(pre + '.out.f64').reshape(2, -1).T if found and os.path.exists(pre + '.out.f64') else None
            for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
            os.rmdir(tmp)
        row = dict(name=nm, effect=m['effect'], A=m['A'], B=m['B'], found=bool(found), params=m['params'], took=time.time() - t0); row.update(J); ws, wh = want(m); row['want_step'] = ws; row['want_hold'] = wh
        if o is not None: nb = min(len(o), len(yb)); row['onto'] = score.removed(yb[:nb], xb[:nb], o[:nb])[0]; row['onto_db'] = db(o[:nb] - yb[:nb], yb[:nb])
        if back is not None: na = min(len(back), len(xa)); row['out'] = score.removed(xa[:na], ya[:na], back[:na])[0]; row['out_db'] = db(back[:na] - xa[:na], xa[:na])
        out.write(json.dumps(row) + '\n'); out.flush()
else:
    def mask(nm): return nm if not nm.startswith('real_') else 'one of the recordings'
    for how in ('py', 'app'):
        fn = HERE + 'bf7_%s.jsonl' % how
        if not os.path.exists(fn): continue
        rows = [json.loads(l) for l in open(fn) if l.strip()]; print('%s: %d fresh crush cases' % (how, len(rows)))
        for r in rows: r['want_step'], r['want_hold'] = want(dict(params=r['params'], effect=r['effect']))
        for e in ('plain', 'bit_pedalboard', 'bit_swh', 'bit_calf', 'bit_mda'):
            q = [r for r in rows if r['effect'] == e]; g = [r for r in q if r['found']]
            if not q: continue
            on = [r['onto'] for r in g if 'onto' in r]; ou = [r['out'] for r in g if 'out' in r]
            st = [abs(np.log2(r['step'][0] / r['want_step'])) for r in g if r.get('want_step') and r.get('step') and r['step'][0] > 0]; ho = [abs(r['P'] - r['want_hold']) for r in g if r.get('want_hold') is not None and r.get('P') is not None]
            print('  %-15s found %2d of %2d; step off by (bits): median %.4f, most %.4f (%d with a grid); hold off by (samples): median %.4f, most %.4f; onto B: median %.1f, least %.1f%s; %.2f s a sound' % (e, len(g), len(q), np.median(st) if st else 0, max(st) if st else 0, len(st), np.median(ho) if ho else 0, max(ho) if ho else 0, np.median(on) if on else 0, min(on) if on else 0, ('; taken out: median %.1f, least %.1f' % (np.median(ou), min(ou))) if ou else '', np.median([r['took'] for r in q])))
            for r in q:
                bad = (not r['found']) or r.get('onto', 100) < 80 or (r.get('want_hold') is not None and r.get('P') is not None and abs(r['P'] - r['want_hold']) > 0.01)
                if bad or os.environ.get('ALL'):
                    p = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r['params'].items() if k in ('bits', 'samples', 'down', 'morph', 'anti_aliasing', 'quant', 'rate', 'post_filt', 'step', 'offset', 'P', 'c')}
                    gr = r.get('grids') or [None]; g0 = gr[0]; h = r.get('hold') or {}
                    print('      %-18s A %-22s %s | found %d grid %s hold P %.4f share %.3f (%d) | onto %.1f (%.1f dB)' % (r['name'], mask(r['A']), p, r['found'], 'none' if g0 is None else 'step 1/%.3f share %.3f' % (1 / g0['step'], g0['share']), h.get('P', 0), h.get('share', 0), h.get('changes', 0), r.get('onto', 0), r.get('onto_db', 0)))
