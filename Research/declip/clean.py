# The declipper on sounds without an effect: the clean corpus clips through the engine's test program, build 0.5.1's
# way (DR_BEND_PLAIN) against the new one. How many show DISTORTION or SATURATION, how many of them by a plain cut
# alone, and what such a clip's top looks like. Prints kinds only, never a clip's name.
#   python3 clean.py [corpus_dir]
import sys, os, glob, json, subprocess, collections, numpy as np, soundfile as sf
C = (sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/v2/corpus').rstrip('/') + '/'; exe = os.environ.get('EXE', '/tmp/claude-0/v2/declip/bin/drbend'); model = '/home/claude/defect-refect/Resources/bend.bin'
meta = json.load(open(C + 'meta.json')); kind = lambda n: 'own' if 'real' in n else str(meta.get(n, {}).get('kind', n.split('_')[0])); rows = []
for f in sorted(glob.glob(C + 'dry/*.wav')):
    n = os.path.basename(f)[:-4]; r = {}
    for tag, env in (('plain', dict(DR_BEND_PLAIN='1')), ('new', {})):
        p = subprocess.run([exe, model, f, '-', '/dev/shm/x40_' + tag, '0'], capture_output=True, text=True, env=dict(os.environ, **env))
        if p.returncode: r[tag] = None; continue
        w = p.stdout.splitlines()[0].split(); fl = [l for l in p.stdout.splitlines() if l.startswith('flat:')]; r[tag] = dict(found=int(w[1]), sure=float(w[5]), top=float(w[7]), kind=w[10], flat=fl[0] if fl else '')
    rows.append((kind(n), r))
for t in ('plain', 'new'):
    for k in ('.out.f64', '.feat.f64', '.bend.f64'):
        if os.path.exists('/dev/shm/x40_' + t + k): os.remove('/dev/shm/x40_' + t + k)
ok = [(k, r) for k, r in rows if r['plain'] and r['new']]; print(len(rows), 'clips,', len(ok), 'read')
shown = lambda d: d['found'] and d['sure'] >= 0.65
print('shown where the slider stands at first: build 0.5.1 %d, new %d; found at all: %d, %d' % (sum(shown(r['plain']) for k, r in ok), sum(shown(r['new']) for k, r in ok), sum(r['plain']['found'] for k, r in ok), sum(r['new']['found'] for k, r in ok)))
g = lambda fl, key: fl.split()[fl.split().index(key) + 1]
print('kinds of top (new):', dict(collections.Counter(g(r['new']['flat'], 'kinds') if r['new']['flat'] else 'not read' for k, r in ok)))
for k, r in ok:
    a, b = r['plain'], r['new']
    if shown(a) != shown(b) or a['found'] != b['found'] or (b['flat'] and g(b['flat'], 'kinds') != 'none/none'):
        print('  %-14s 0.5.1: found %d sure %.2f top %5.2f %-10s | new: found %d sure %.2f top %5.2f %-10s | %s' % (k, a['found'], a['sure'], a['top'], a['kind'], b['found'], b['sure'], b['top'], b['kind'], b['flat'][6:150]))
