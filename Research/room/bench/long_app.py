# The app itself (Tests/render.cpp built as drrender, at the default Sureness) on the long sounds of prep_long.py: is a
# REVERB group shown? Unlike long_eval3.py (the room engine's verdict alone) this includes the look without the other
# found effects (Estimate.h, recheckRoomWithout).
#   python3 long_app.py path/to/drrender path/to/Resources tag part of     -> long_app_<tag>_<part>.jsonl (resumable)
#   python3 long_app.py --show tag [tag2]
import sys, os, json, glob, re, subprocess, tempfile, collections
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'; D = '/tmp/claude-0/v2/roomsure/'
def name(r):
    c = int(r['cut'])
    if r['kind'] == 'loopc': return 'clean clip looped'
    if r['kind'] == 'catc': return 'clips in a row, clean'
    if r['kind'] == 'catfx': return 'clips in a row with sweep, phaser or chorus'
    return 'reverb, rings out' if not c else 'reverb, cut off'
def show(tags):
    for t in tags:
        R = [json.loads(l) for f in sorted(glob.glob(D + 'long_app_%s_*.jsonl' % t)) for l in open(f)]; print('[%s] %d looks' % (t, len(R)))
        G = collections.OrderedDict()
        for r in R: G.setdefault(name(r), []).append(r)
        for lab, g in list(sorted(G.items())) + [('ALL without reverb', [r for r in R if not r['wet']]), ('ALL with reverb', [r for r in R if r['wet']])]:
            if g: print('  %-46s n=%3d | room found on the whole sound %3d | the engine\'s verdict over 65 %%: %3d (%2.0f %%) | REVERB shown %3d (%2.0f %%) | looked at again without other effects %3d, of them lost %d' % (
                lab, len(g), sum(r['found'] for r in g), sum(r['sure'] > 0.65 for r in g), 100 * np.mean([r['sure'] > 0.65 for r in g]), sum(r['shown'] for r in g), 100 * np.mean([r['shown'] for r in g]),
                sum(r['again'] is not None for r in g), sum(r['again'] is not None and r['sure'] > 0.65 and not r['shown'] for r in g)))
        print('  other groups shown on the sounds: ' + ', '.join('%s %d' % kv for kv in collections.Counter(o for r in R for o in r['others']).most_common()))
if sys.argv[1] == '--show': show(sys.argv[2:]); sys.exit(0)
EXE, RES, TAG = sys.argv[1:4]; part, of = int(sys.argv[4]), int(sys.argv[5])
OUT = D + 'long_app_%s_%d.jsonl' % (TAG, part); done = {(json.loads(l)['path'], json.loads(l)['cut']) for l in open(OUT)} if os.path.exists(OUT) else set(); tmp = tempfile.mkdtemp()
rows = json.load(open(D + 'long_rows.json')); want = set(a for a in sys.argv[6:])        # kinds/cuts to look at: e.g. "loopc:0" "loopr:18"; none: all
for r in rows[part::of]:
    key = '%s:%d' % (r['kind'], int(r['cut']))
    if (r['path'], int(r['cut'])) in done or (want and key not in want and r['kind'] not in want): continue
    pin = r['path']
    if int(r['cut']): y, sr = sf.read(r['path'], dtype='float64', always_2d=True, frames=int(r['cut']) * 48000); pin = os.path.join(tmp, 'in.wav'); sf.write(pin, y, sr, subtype='FLOAT')
    q = subprocess.run([EXE, RES, pin, '-', '-'], capture_output=True, text=True)
    if q.returncode: raise RuntimeError(q.stdout[-300:] + q.stderr[-300:])
    # with no output file the tool prints what it found and every effect's sureness on one line ("autopan=0.03 ... reverb=0.94 ..."); a group is shown above 65 %
    rm = re.search(r'^room: found=(\d) .*? sureness=([0-9.eE+-]+)', q.stdout, re.M); ag = re.search(r'^room without the other effects \(([^)]*)\): sureness=([0-9.eE+-]+) counts=([0-9.eE+-]+)', q.stdout, re.M)
    pr = [l for l in q.stdout.splitlines() if l.startswith('autopan=') and ' reverb=' in l]; probs = dict((a, float(b)) for a, b in re.findall(r'([a-z0-9]+)=([0-9.eE+-]+)', pr[-1])) if pr else {}
    found = bool(int(rm.group(1))) if rm else False
    o = dict(path=r['path'], cut=int(r['cut']), kind=r['kind'], wet=r['wet'], split=r['split'], tool=r['tool'], found=found, sure=float(rm.group(2)) if rm else 0.0,
             again=(dict(what=ag.group(1), sure=float(ag.group(2)), counts=float(ag.group(3))) if ag else None), shown=bool(found and probs.get('reverb', 0.0) > 0.65), prob=probs.get('reverb'), others=sorted(k for k, v in probs.items() if k != 'reverb' and v > 0.65))
    open(OUT, 'a').write(json.dumps(o) + '\n'); print(os.path.basename(r['path']), o['cut'], 'found %d sure %.2f again %s shown %d others %s' % (o['found'], o['sure'], ('%.2f' % o['again']['sure']) if o['again'] else '-', o['shown'], ','.join(o['others'])), flush=True)
