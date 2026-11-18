# The look "is the room still there with the other found effects taken out" (Estimate.h, recheckRoomWithout), measured
# with the app's own processing (Tests/render.cpp built as drrender) on the test sounds of prep_nw.py and prep_sb.py.
# Only sounds on which the room engine is at least 50 % sure are run (the look can only lower a verdict).
#   python3 recheck_eval.py path/to/drrender path/to/Resources tag neg|pos [step] [part of]   -> recheck_<tag>_<neg|pos>_<part>.jsonl (resumable)
#   python3 recheck_eval.py --show tag
# neg = sounds without reverb (clean clips, other effects alone, tone effects alone), pos = sounds with reverb (alone or after
# another effect, ringing out or cut off at 6 s). Per sound: the room engine's sureness on the sound as it is, which groups
# were taken out, the sureness without them, and every effect's score.
import sys, os, json, re, glob, subprocess, tempfile, collections, numpy as np
sys.path.insert(0, '/tmp/claude-0/v2/roomsure'); import look2
D = '/tmp/claude-0/v2/roomsure/'; BASE = os.environ.get('BASE', 'x45')
def kind(r):
    if r.get('set') == 'nw': return {'fx': 'effect only', 'rev': 'reverb', 'fxr': 'effect + reverb'}[r['kind']] + (', cut off' if int(r.get('cut', 0)) else ', rings out' if r['kind'] != 'fx' else ', cut off')
    return {'clean': 'clean, cut off', 'reverb': 'reverb, rings out', 'tone': 'tone only'}.get(r['fam'], 'effect only, cut off')
def rows_all():
    rs = look2.load('nw', BASE) + look2.load('sb', BASE)
    for r in rs: r['key'] = '%s|%d' % (os.path.basename(r['path']), int(r.get('cut', 0)))
    return rs
def show(tag):
    rs = rows_all(); out = {}
    for f in sorted(glob.glob(D + 'recheck_%s_*.jsonl' % tag)):
        for l in open(f): j = json.loads(l); out[j['key']] = j
    print('%d sounds run of %d that are at least 50 %% sure' % (len(out), sum(r['v_as'] > 0.5 for r in rs)))
    def after(r): j = out.get(r['key']); return (j['final'] if j else None) if r['v_as'] > 0.5 else r['v_as']
    K = collections.OrderedDict()
    for r in rs: K.setdefault((kind(r), r['split']), []).append(r)
    print('%-28s %-5s %5s %5s | shown at 65%%: before  after | at 50%%: before  after | sounds looked at again, groups taken out' % ('kind', 'split', 'n', 'run'))
    for (k, sp), g in sorted(K.items()):
        run = [r for r in g if after(r) is not None]; a = np.array([after(r) for r in run]); b = np.array([r['v_as'] for r in run]); took = collections.Counter(w for r in run if r['key'] in out and out[r['key']]['what'] for w in out[r['key']]['what'].split(','))
        again = sum(1 for r in run if r['key'] in out and out[r['key']]['what'])
        print('%-28s %-5s %5d %5d | %19.1f%% %5.1f%% | %14.1f%% %5.1f%% | %d: %s' % (k, sp, len(g), len(run), 100 * np.mean(b > 0.65), 100 * np.mean(a > 0.65), 100 * np.mean(b > 0.5), 100 * np.mean(a > 0.5), again, ', '.join('%s %d' % kv for kv in took.most_common())))
    for lab, sel in (('without reverb', lambda r: r['wet'] == 0), ('with reverb', lambda r: r['wet'] == 1)):
        for sp in ('dev', 'test'):
            run = [r for r in rs if sel(r) and r['split'] == sp and after(r) is not None]; a = np.array([after(r) for r in run]); b = np.array([r['v_as'] for r in run])
            print('all %-15s %-4s n=%4d | shown at 65%%: before %5.1f%% after %5.1f%%' % (lab, sp, len(run), 100 * np.mean(b > 0.65), 100 * np.mean(a > 0.65)))
    fam = collections.defaultdict(lambda: [0, 0, 0])
    for r in rs:
        if r['wet'] == 0 and r['fam'] not in ('clean', 'tone') and after(r) is not None: f = fam[r['fam']]; f[0] += 1; f[1] += r['v_as'] > 0.65; f[2] += after(r) > 0.65
    print('effect alone, REVERB shown before -> after, by family: ' + ', '.join('%s %d -> %d (of %d)' % (k, v[1], v[2], v[0]) for k, v in sorted(fam.items(), key=lambda kv: -kv[1][1])))
    # sounds with reverb that lose their group to the look: what was taken out
    lost = collections.Counter(); kept = collections.Counter()
    for r in rs:
        j = out.get(r['key'])
        if r['wet'] == 1 and j and j['what'] and r['v_as'] > 0.65: (lost if j['final'] <= 0.65 else kept)[j['what']] += 1
    print('sounds with reverb looked at again: group kept %d, lost %d | lost by what was taken out: %s | kept: %s' % (sum(kept.values()), sum(lost.values()), dict(lost.most_common()), dict(kept.most_common(8))))
if sys.argv[1] == '--show': show(sys.argv[2]); sys.exit(0)
EXE, RES, TAG, WHICH = sys.argv[1:5]; step = int(sys.argv[5]) if len(sys.argv) > 5 else 1; part, of = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (0, 1)
import soundfile as sf
OUT = D + 'recheck_%s_%s_%d.jsonl' % (TAG, WHICH, part); done = {json.loads(l)['key'] for l in open(OUT)} if os.path.exists(OUT) else set(); tmp = tempfile.mkdtemp()
sel = [r for r in rows_all() if r['v_as'] > 0.5 and (r['wet'] == 1) == (WHICH == 'pos')][::step][part::of]
for r in sel:
    if r['key'] in done: continue
    p = r['path']; c = int(r.get('cut', 0))
    if c:
        y, sr = sf.read(p, dtype='float64', always_2d=True); p = os.path.join(tmp, 'cut.wav'); sf.write(p, y[:c * sr], sr, subtype='PCM_16')
    q = subprocess.run([EXE, RES, p, '-', '-'], capture_output=True, text=True)
    if q.returncode: print('FAILED', r['key'], q.stderr[-200:], flush=True); continue
    m0 = re.search(r'^room: found=(\d) .*? sureness=([0-9.eE+-]+)', q.stdout, re.M); m1 = re.search(r'^room without the other effects \(([a-z,]*)\): sureness=([0-9.eE+-]+) counts=([0-9.eE+-]+)', q.stdout, re.M)
    probs = dict((a, float(b)) for a, b in re.findall(r'([a-z0-9]+)=([0-9.eE+-]+) ', q.stdout.strip().split('\n')[-1] + ' '))
    j = dict(key=r['key'], s0=float(m0.group(2)), found=int(m0.group(1)), what=m1.group(1) if m1 else '', s1=float(m1.group(2)) if m1 else None, final=probs.get('reverb'), probs={k: round(v, 3) for k, v in probs.items() if v > 0.3})
    open(OUT, 'a').write(json.dumps(j) + '\n'); print('%-44s as is %.2f (study %.2f) | without %-22s %s | counts %.2f' % (r['key'], j['s0'], r['v_as'], j['what'] or '-', '%.2f' % j['s1'] if m1 else '  - ', j['final']), flush=True)
