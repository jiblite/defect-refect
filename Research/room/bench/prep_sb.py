# The scoreboard's own sounds as WAV files for rsure: every family's wet A and B, and the corpus's clean clips.
import glob, os, json, soundfile as sf
OUT = '/tmp/claude-0/v2/roomsure/wav/sb/'; rows = []
for d in sorted(glob.glob('/tmp/claude-0/v2/bench/sets/*/*/')):
    fam, case = d.rstrip('/').split('/')[-2:]; m = json.load(open(d + 'meta.json'))
    for w in 'AB':
        p = OUT + '%s__%s__%s.wav' % (fam, case, w)
        if not os.path.exists(p):
            x, sr = sf.read(d + w + '_wet.flac', always_2d=True); sf.write(p, x, sr, subtype='PCM_16')
        rows.append(dict(path=p, set='sb', fam=fam, case=case, which=w, tool=m['effect'], split=m.get('split', ''), dry=m[w], wet=int(fam == 'reverb')))
cm = json.load(open('/tmp/claude-0/v2/corpus/meta.json'))
for k, v in sorted(cm.items()): rows.append(dict(path='/tmp/claude-0/v2/corpus/dry/%s.wav' % k, set='sb', fam='clean', case=k, which='', tool='none', split=v['split'], dry=k, wet=0, kind=v['kind']))
json.dump(rows, open('sb_rows.json', 'w')); open('sb_list.txt', 'w').write('\n'.join(r['path'] for r in rows) + '\n'); print(len(rows))
