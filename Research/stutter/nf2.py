# The new engines in the app's own code (drstutter, drcrush: "find") on every clean clip and on every scoreboard sound,
# against what the prototypes said (nf1_*.jsonl).   python3 nf2.py stutter|crush [clean|fams]
import sys, os, json, glob, subprocess, tempfile
sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
eng = sys.argv[1]; BIN = '/tmp/claude-0/dr' + eng; HERE = '/tmp/claude-0/v2/newfx/'
for what in (sys.argv[2:] or ['clean', 'fams']):
    rows = [json.loads(l) for l in open(HERE + 'nf1_%s.jsonl' % what) if l.strip()]; tmp = tempfile.mkdtemp(dir='/dev/shm'); pa = tmp + '/a.wav'; out = open(HERE + 'nf2_%s_%s.jsonl' % (eng, what), 'w'); diff = 0; found = 0; took = []
    if what == 'clean': paths = {r['name']: '/tmp/claude-0/v2/corpus/dry/' + r['name'] + '.wav' for r in rows}
    else: paths = {r['name']: SETS + r['name'].split(':')[0] + '/' + r['name'].split(':')[1] + '_wet.flac' for r in rows}
    fam = dict(crush='bitcrush', stutter='stutter')[eng]; own = 0; ownn = 0
    for r in rows:
        y = truth.rd(paths[r['name']]); sf.write(pa, y, SR, subtype='FLOAT'); p = subprocess.run([BIN, 'find', pa], capture_output=True, text=True)
        if p.returncode: print('ERROR', r['name'] if 'real_' not in r['name'] else 'one of the recordings', p.stderr[-200:]); diff += 1; continue
        J = json.loads(p.stdout.strip().splitlines()[-1]); took.append(J['took']); out.write(json.dumps(dict(name=r['name'], **J)) + '\n'); mine = r['name'].startswith(fam + '/'); ownn += mine; own += mine and J['found']; found += J['found'] and not mine
        if eng == 'stutter': same = J['found'] == r['stutter']['found'] and J['events'] == [list(e) for e in r['stutter']['events']]
        else: same = J['found'] == r['crush']['found'] and (not J['found'] or (abs(J['bits'] - r['crush']['bits']) < 1e-3 and abs(J['P'] - r['crush']['P']) < 1e-6))
        if not same: diff += 1; print('   differs: %s | prototype %s | app %s' % (r['name'] if 'real_' not in r['name'] else 'one of the recordings', str(r[eng])[:200], str({k: v for k, v in J.items() if k not in ('stretches',)})[:300]))
    os.remove(pa); os.rmdir(tmp)
    print('%s, %s: %d sounds, found on %d others%s, %d differ from the prototype; seconds a sound: median %.3f, most %.3f' % (eng, what, len(rows), found, (' and on %d of its own %d' % (own, ownn)) if ownn else '', diff, np.median(took), max(took)))
