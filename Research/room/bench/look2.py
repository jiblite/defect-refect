# What the room engine says on the new test sounds (nw) and on the scoreboard's own (sb), by kind of sound.
#   python3 look2.py tag [tag2]       (tag: the run of rsure, e.g. 044 = the models of build 0.4.4; two tags: side by side)
# shown = the app's rule: sureness over 0.65, where the sureness of a found room is the smaller of the two looks' (the sound
# as it is, and with 3 s of silence after it). "as is" = the first look alone.
import sys, os, json, collections, numpy as np
D = '/tmp/claude-0/v2/roomsure/'
def load(name, tag):
    rows = json.load(open(D + name + '_rows.json')); out = {}
    try:
        for l in open(D + '%s_out_%s.jsonl' % (name, tag)):
            j = json.loads(l); out[(os.path.basename(j['path']), int(j.get('cut', 0)))] = j         # by file name: the folder has moved since the first runs
    except FileNotFoundError: return []
    for r in rows:
        j = out.get((os.path.basename(r['path']), int(r.get('cut', 0))))
        if j is None: continue
        r['j'] = j; r['v_as'] = j['as']['sure'] if j['as']['found'] else 0.09; r['v'] = min(j['as']['sure'], j['end']['sure']) if j['as']['found'] else 0.09; r['v_end'] = j['end']['sure'] if j['end']['found'] else 0.09
    return [r for r in rows if 'j' in r]
def groups(nw, sb):
    G = collections.OrderedDict()
    def add(k, r): G.setdefault(k, []).append(r)
    for r in sb:
        if r['fam'] == 'clean': add(('clean clip, cut off', r['split']), r)
        elif r['fam'] == 'reverb': add(('scoreboard reverb, rings out', r['split']), r)
        elif r['fam'] == 'tone': add(('scoreboard tone only', r['split']), r)
        else: add(('scoreboard other effect', r['split']), r)
    for r in nw:
        k = r['kind']; c = int(r.get('cut', 0))
        if k == 'fx': add(('other effect, new draw', r['split']), r)
        elif k == 'rev': add(('reverb, ' + ('cut off' if c else 'rings out'), r['split']), r)
        else: add(('effect + reverb, ' + ('cut off' if c else 'rings out'), r['split']), r)
    return G
def line(rs, cut=0.65): return (len(rs), 100 * np.mean([r['j']['as']['found'] for r in rs]), 100 * np.mean([r['v_as'] > cut for r in rs]), 100 * np.mean([r['v_end'] > cut for r in rs]), 100 * np.mean([r['v'] > cut for r in rs]))
if __name__ == '__main__':
    tags = sys.argv[1:] or ['044']; data = {t: groups(load('nw', t), load('sb', t)) for t in tags}
    print('%-34s %-5s %4s | %s' % ('kind of sound', 'split', 'n', ' | '.join('%-44s' % ('[%s] found, shown: as is / with silence / both' % t) for t in tags)))
    keys = list(data[tags[0]].keys())
    for k in sorted(keys):
        print('%-34s %-5s %4d | %s' % (k[0], k[1], len(data[tags[0]][k]), ' | '.join('%5.0f%%   %5.0f%% / %5.0f%% / %5.0f%%              ' % line(data[t][k])[1:] if k in data[t] else '-' for t in tags)))
    # the other effects by family (new draws + scoreboard): shown by the app's rule
    for t in tags:
        fam = collections.defaultdict(list)
        for k, rs in data[t].items():
            if k[0] in ('other effect, new draw', 'scoreboard other effect'):
                for r in rs: fam[r['fam']].append(r)
        print('[%s] other effects, REVERB shown (both looks): ' % t + ', '.join('%s %d/%d' % (f, sum(r['v'] > 0.65 for r in rs), len(rs)) for f, rs in sorted(fam.items())) + ' | all %d/%d' % (sum(r['v'] > 0.65 for rs in fam.values() for r in rs), sum(len(rs) for rs in fam.values())))
