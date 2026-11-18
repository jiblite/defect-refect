# What the app's chorus engine (drchorus find) says on the clean corpus, on every wet sound of the scoreboard's families and
# on the extra chorus cases. For the sureness rule. One line of JSON per sound.
#   python3 cfind.py out.jsonl clean | x | family ...          python3 cfind.py --show out.jsonl ...
import sys, os, json, subprocess, numpy as np
sys.path[:0] = ['/tmp/claude-0/v2/chorus', '/tmp/claude-0/v2/sweep']
EXE = os.environ.get('EXE', '/tmp/claude-0/lvx/drchorus'); TMP = '/dev/shm/chf/'
def voices(r): return [v for pc in r['pieces'] for side in pc['voices'] for v in side]
def show(paths):
    R = [json.loads(l) for p in paths for l in open(p)]; fams = sorted({r['fam'] for r in R}, key=lambda f: (f != 'clean', f != 'chorus', f != 'x', f))
    for f in fams:
        rows = [r for r in R if r['fam'] == f]; on = [r for r in rows if r['found']]; over = [r for r in on if r['sure'] > 0.65]
        print('%-12s %3d sounds | found %3d, over 0.65: %3d | sure of the found: %s | %.1f s a sound' % (f, len(rows), len(on), len(over), ' '.join('%.2f' % r['sure'] for r in sorted(on, key=lambda r: -r['sure']))[:100], np.mean([r['took'] for r in rows])))
        for r in sorted(on, key=lambda r: -r['sure'])[:int(os.environ.get('TOP', 6))]:
            vs = sorted(voices(r), key=lambda v: -v['clear'])[:3]
            print('      %-22s sure %.2f  %s' % (r['name'] if 'real' not in r['name'] else '(a recording)', r['sure'], ' | '.join('%s %.3f Hz %.2f+-%.2f ms on %.2f loud %.2f gone %.2f least %.2f twin %.2f' % ('tri' if v['kind'] else 'sin', v['rate'], v['mid'] / 24.0, abs(v['half']) / 24.0, v['on'], v['loud'], v['after'] / max(v['before'], 1e-9), v['least'], v['twin']) for v in vs)))
        if os.environ.get('NOTES'):
            for r in rows:
                for n in r['notes']:
                    if 'no voice' in n or 'not used' in n: print('      . %-20s %s' % (r['name'] if 'real' not in r['name'] else '(a recording)', n))
if __name__ == '__main__':
    if sys.argv[1] == '--show': show(sys.argv[2:]); sys.exit()
    import soundfile as sf
    from lab import cases, load
    os.makedirs(TMP, exist_ok=True); out = sys.argv[1]; done = {json.loads(l)['name'] for l in open(out)} if os.path.exists(out) else set(); f = open(out, 'a')
    def one(fam, name, x):
        if name in done: return
        p = TMP + 'in_%d.wav' % os.getpid(); sf.write(p, x, 48000, subtype='FLOAT'); r = subprocess.run([EXE, 'find', p], capture_output=True, text=True)
        try: j = json.loads(r.stdout.strip().split('\n')[-1])
        except Exception: print('FAILED', fam, name, r.stderr[-300:], flush=True); return
        j.update(fam=fam, name=name); f.write(json.dumps(j) + '\n'); f.flush()
    for a in sys.argv[2:]:
        if a == 'clean':
            meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
            for cid in sorted(meta):
                x = sf.read('/tmp/claude-0/v2/corpus/dry/%s.wav' % cid, dtype='float64', always_2d=True)[0]; x = x if x.shape[1] > 1 else np.repeat(x, 2, 1); one('clean', cid, x[:, :2])
        elif a == 'x':
            cs = json.load(open('/dev/shm/chx/cases.json'))
            for name in sorted(cs):
                for w in 'AB': one('x', name + '_' + w, sf.read('/dev/shm/chx/%s_%s_wet.wav' % (name, w), dtype='float64', always_2d=True)[0])
        else:
            for d in cases(a):
                for w in 'AB': c = load(d, w); one(a, c['name'] + '_' + w, c['wet'])
        print(a, flush=True)
