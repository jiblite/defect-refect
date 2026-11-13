# What the app itself does on the scoreboard's cases of every effect family except reverb and tone (those have their own
# evaluations). Runs the app's processing without the window (Tests/render.cpp, built as drrender) and scores against the truth.
#   python3 famboard.py path/to/drrender path/to/Resources [family ...]        results: famboard.jsonl (one line per case, resumable)
#   python3 famboard.py --show [file]                                           prints the table
# Per case (sound A through an outside effect, sound B through the same effect):
#   found      is the family's own group shown at the default Sureness, and which other groups are shown
#   out        one sound: the family's group at Amount far left, the others left alone   (100 = the dry sound, 0 = nothing gained)
#   out_all    one sound: every shown group at Amount far left
#   onto       two sounds: only the family's group put on dry B                          (100 = B through the real effect, 0 = dry B)
#   onto_all   two sounds: everything the app shows put on dry B (what a user gets by default)
import sys, os, json, subprocess, tempfile, re
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'proto'))
SETS = os.path.join(HERE, 'sets'); SKIP = ('reverb', 'tone'); SR = 48000

def show(path):
    rows = [json.loads(l) for l in open(path)]; fams = sorted({r['fam'] for r in rows})
    print('%-12s %2s  %5s  %-23s %-23s %-23s %-23s  %s' % ('family', 'n', 'found', 'out (own group)', 'out (all shown)', 'onto (own group)', 'onto (all shown)', 'others shown per case'))
    def st(v):
        v = np.array([x for x in v if x is not None], float)
        return '%5.0f %5.0f %2d%% (%d)' % (np.median(v), v.mean(), round(100 * (v < 0).mean()), len(v)) if len(v) else '    -     -   - (0)'
    tot = dict(out=[], out_all=[], onto=[], onto_all=[])
    for f in fams:
        rs = [r for r in rows if r['fam'] == f]
        print('%-12s %2d  %2d/%-2d  %-23s %-23s %-23s %-23s  %.1f' % (f, len(rs), sum(r['found'] for r in rs), len(rs), st([r['out'] for r in rs]), st([r['out_all'] for r in rs]),
              st([r['onto'] for r in rs]), st([r['onto_all'] for r in rs]), np.mean([len(r['others']) for r in rs])))
        for k in tot: tot[k] += [r[k] for r in rs]
    print('%-12s %2d  %2d/%-2d  %-23s %-23s %-23s %-23s' % ('all', len(rows), sum(r['found'] for r in rows), len(rows), st(tot['out']), st(tot['out_all']), st(tot['onto']), st(tot['onto_all'])))
    print('columns: median, mean, share of cases made worse than doing nothing, (cases). A case where the group is not shown counts as 0 (nothing done).')

if len(sys.argv) > 1 and sys.argv[1] == '--show': show(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'famboard.jsonl')); sys.exit(0)
import truth, score
EXE, RES = sys.argv[1], sys.argv[2]; want = sys.argv[3:]; OUT = os.environ.get('OUT', os.path.join(HERE, 'famboard.jsonl'))
done = {json.loads(l)['name'] for l in open(OUT)} if os.path.exists(OUT) else set()
tmp = tempfile.mkdtemp()
ALL = 'flanger chorus phaser haas width delay reverb distortion bitcrush filtersweep tremolo autopan saturation exciter ringmod filter stutter gatedrhythm comb'.split()
def run(first, blocks):
    """One analysis of the first sound, several renders (blocks: (second or '-', out, [fader=value ...])). Returns the groups the app shows."""
    args = []
    for i, (second, out, fad) in enumerate(blocks): args += (['--'] if i else []) + [second, out] + list(fad)
    r = subprocess.run([EXE, RES, first] + args, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
    groups = {}
    for l in r.stdout.split('\n'):
        m = re.match(r'^([a-z]+) p=([0-9.eE+-]+)', l)
        if m: groups[m.group(1)] = float(m.group(2))
    return groups
def rd(p, n):
    y, _ = sf.read(p, dtype='float64', always_2d=True); return y[:n]
for fam in sorted(os.listdir(SETS)):
    if fam in SKIP or (want and fam not in want): continue
    for name in sorted(os.listdir(os.path.join(SETS, fam))):
        if name in done: continue
        d = os.path.join(SETS, fam, name) + '/'; cA = truth.load(d, 'A'); cB = truth.load(d, 'B')
        pa, pb = os.path.join(tmp, 'a.wav'), os.path.join(tmp, 'b.wav'); po = [os.path.join(tmp, 'o%d.wav' % i) for i in range(4)]
        sf.write(pa, cA['wet'], SR, subtype='FLOAT'); sf.write(pb, cB['dry'], SR, subtype='FLOAT')
        g = run(pa, [('-', po[0], [fam + '.amount=0']), ('-', po[1], [k + '.amount=0' for k in ALL]), (pb, po[2], []), (pb, po[3], [k + '.amount=0' for k in ALL if k != fam])])
        found = fam in g; others = sorted(k for k in g if k != fam); nA, nB = len(cA['wet']), len(cB['dry'])
        row = dict(fam=fam, name=name, effect=cA['meta']['effect'], found=found, p=g.get(fam), others=others, shown=g)
        row['out'] = score.removed(cA['dry'], cA['wet'], rd(po[0], nA))[0] if found else 0.0
        row['out_all'] = score.removed(cA['dry'], cA['wet'], rd(po[1], nA))[0] if g else 0.0
        row['onto_all'] = score.removed(cB['wet'], cB['dry'], rd(po[2], nB))[0] if g else 0.0
        row['onto'] = score.removed(cB['wet'], cB['dry'], rd(po[3], nB))[0] if found else 0.0
        open(OUT, 'a').write(json.dumps(row) + '\n')
        print('%-12s %-26s found %d (p %s) out %6.1f out_all %6.1f onto %6.1f onto_all %6.1f others %s' % (fam, name, found, '%.2f' % g[fam] if found else '-', row['out'], row['out_all'], row['onto'], row['onto_all'], ' '.join(others)), flush=True)
show(OUT)
