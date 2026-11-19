# The stereo engine's reference (stereo.py) on the scoreboard cases, on the clean clips and on the other families' sounds.
#   python3 st5.py board [family ...]      scores (removal at far left, half way, onto the case's other sound)
#   python3 st5.py clean i n               the numbers behind the verdict on the clean clips i::n      -> st5_clean_<i>.jsonl
#   python3 st5.py fams i n [family ...]   the same on every scoreboard sound (A and B) of the families  -> st5_fams_<i>.jsonl
import sys, json, time; sys.path.insert(0, '/tmp/claude-0/v2/newfx')
from st_common import *
import sides as E
def numbers(f):
    r = dict(kind=f['kind'], mode=f['mode'])
    if 'pair' in f:
        p, sh, ck = f['pair'], f['shape'], f['pair_check']; la = np.where(p['a'] != 0)[0]; lb = np.where(p['b'] != 0)[0]
        r.update(way=p['way'], res=float(10 * np.log10(max(p['res'], 1e-30))), fits=p['fits'], taps=sh['taps'], apart=sh['apart'], reach=sh['reach'], made=sh['made'], covered=ck['covered'], inside=float(10 * np.log10(max(ck['inside'], 1e-30))), beyond=ck['beyond'],
                 a=[(int(i), float(p['a'][i])) for i in la][:12], b=[(int(i), float(p['b'][i])) for i in lb][:12], spans=len(ck['spans']))
    if 'copy' in f:
        g = f['copy']; r.update(share=g['share'], share_all=g['all'], balance=g['balance'], side_db=g['side_db'])
        if 'copy_check' in f: r.update(copy_covered=f['copy_check']['covered'], copy_inside=float(10 * np.log10(max(f['copy_check']['inside'], 1e-30))))
    return r
def taps(v): return ' '.join('%+.3f@%.2fms' % (x, i / 48.0) for i, x in v[:6]) + (' ...' if len(v) > 6 else '')
if __name__ == "__main__" and sys.argv[1] == "board":
    for fam in sys.argv[2:] or ('haas', 'width'):
        for d in cases(fam):
            c = truth.load(d, 'A'); cb = truth.load(d, 'B'); dry, wet = c['dry'], c['wet']; t0 = time.time(); f = E.find(wet); t1 = time.time(); r = numbers(f)
            line = '%-24s %-5s %-4s' % (c['name'], f['kind'], f['mode'])
            if 'taps' in r: line += ' | pair: taps %d apart %4d reach %4d made %5.1f dB covered %3.0f %% inside %6.1f dB' % (r['taps'], r['apart'], r['reach'], r['made'], 100 * r['covered'], r['inside'])
            if 'share' in r: line += ' | copy: share %3.0f %% side %5.1f dB' % (100 * r['share'], r['side_db']) + (' covered %3.0f %%' % (100 * r['copy_covered']) if 'copy_covered' in r else '')
            line += ' | %.1f s' % (t1 - t0)
            if f['mode'] is not None:
                o0 = E.own(wet, f, 0.0); oh = E.own(wet, f, 0.5); o1 = E.own(wet, f, 1.0); o2 = E.own(wet, f, 2.0); on = E.onto(cb['dry'], f, 1.0)
                line += '\n      out %4.0f /%4.0f  half way %4.0f /%4.0f  double %4.0f | untouched at the middle: %s | onto %4.0f /%4.0f' % (score.removed(dry, wet, o0)[0], removed4(dry, wet, o0)[0], score.removed(dry, wet, oh)[0], removed4(dry, wet, oh)[0],
                        removed4(dry, wet, o2)[0], 'yes' if np.abs(o1 - wet).max() == 0 else 'NO', score.removed(cb['wet'], cb['dry'], on)[0], removed4(cb['wet'], cb['dry'], on)[0])
            print(line)
            if 'a' in r: print('      a: %s\n      b: %s' % (taps(r['a']), taps(r['b'])), flush=True)
elif __name__ == "__main__":
    i0, n0 = int(sys.argv[2]), int(sys.argv[3])
    if sys.argv[1] == 'clean':
        meta = json.load(open('/tmp/claude-0/v2/corpus/meta.json')); items = [(nm, meta[nm]['kind'], '/tmp/claude-0/v2/corpus/dry/' + nm + '.wav') for nm in sorted(meta)]
    else:
        fams = sys.argv[4:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(SETS + '*/') if 'reverb' not in p)
        items = [(os.path.basename(os.path.dirname(d)) + ':' + w, fam, d + w + '_wet.flac') for fam in fams for d in cases(fam) for w in 'AB']
    out = open('st5_%s_%d.jsonl' % (sys.argv[1], i0), 'w')
    for nm, kind, path in items[i0::n0]:
        x = truth.rd(path); m, s = ms(x); row = dict(name=nm, kind=kind, side_db=db(s, m) if (m ** 2).sum() > 0 else 99.0, secs=0.0)
        if (s ** 2).sum() > 1e-10 * (m ** 2).sum(): t0 = time.time(); f = E.find(x); row.update(numbers(f)); row['found'] = f['kind']; row['secs'] = time.time() - t0
        out.write(json.dumps(row) + '\n'); out.flush()
