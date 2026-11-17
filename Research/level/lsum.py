# One line of numbers per set and family from the long-sound result files (for the write-ups):
#   python3 lsum.py tag            reads long_<set>_<tag>.jsonl for the sets that exist
import sys, os, json, numpy as np
import long as L
HERE = os.path.dirname(os.path.abspath(__file__))
def rows(mode, tag):
    p = os.path.join(HERE, 'long_%s_%s.jsonl' % (mode, tag))
    return sorted((json.loads(l) for l in open(p)), key=lambda r: r['i']) if os.path.exists(p) else None
def fam_line(rs):
    ok = [r for r in rs if L.right(r)]; out = np.array([r['out'] if L.right(r) else 0.0 for r in rs]); onto = np.array([r['onto'] if L.right(r) else 0.0 for r in rs])
    fo = np.array([r['out'] for r in ok]) if ok else np.zeros(1); fn = np.array([r['onto'] for r in ok]) if ok else np.zeros(1)
    lower = sum(r['found'] and not (r['score'] >= 0) for r in rs); wrong = sum(r['found'] and r['score'] >= 0 and not L.right(r) for r in rs)
    return dict(n=len(rs), shown=len(ok), lower=lower, wrong=wrong, out_med=float(np.median(out)), out_shown=float(np.median(fo)), out_mean_shown=float(fo.mean()), out_worse=int((out < 0).sum()),
                onto_med=float(np.median(onto)), onto_shown=float(np.median(fn)), onto_worse=int((onto < 0).sum()), took=float(np.mean([r['took'] for r in rs])), secs=float(np.mean([r['took'] / r['seconds'] for r in rs])))
def summary(tag):
    S = {}
    for mode in ('clean', 'wet', 'part', 'drift', 'slow'):
        rs = rows(mode, tag)
        if rs is None: continue
        if mode == 'clean':
            S[mode] = dict(n=len(rs), shown=sum(r['found'] and r['score'] >= 0 for r in rs), lowest=sum(r['found'] for r in rs), took=float(np.mean([r['took'] for r in rs])), secs=float(np.mean([r['took'] / r['seconds'] for r in rs]))); continue
        S[mode] = {fam: fam_line([r for r in rs if fam == 'all' or r['fam'] == fam]) for fam in ('tremolo', 'autopan', 'gatedrhythm', 'all') if fam == 'all' or any(r['fam'] == fam for r in rs)}
        if mode == 'part':
            off = [r['out_off'] for r in rs if L.right(r) and 'out_off' in r]; S[mode]['off'] = dict(n=len(off), exact=sum(v < -150 for v in off), rest=sorted(round(v) for v in off if v >= -150))
    return S
if __name__ == '__main__':
    for tag in sys.argv[1:]:
        S = summary(tag); print('==', tag)
        for mode, d in S.items():
            if mode == 'clean': print('  clean: %d sounds, shown %d, at the lowest Sureness %d, %.1f s a sound, %.2f s per second of sound' % (d['n'], d['shown'], d['lowest'], d['took'], d['secs'])); continue
            for fam, f in d.items():
                if fam == 'off': print('  %-5s the half without the effect: exact (under -150 dB) in %d of %d shown, the rest at %s dB' % (mode, f['exact'], f['n'], f['rest'])); continue
                print('  %-5s %-11s shown %2d/%-2d (lower Sureness %d, other kind %d) | out median %5.1f, where shown %5.1f (mean %5.1f), worse %d | onto median %5.1f, where shown %5.1f, worse %d | %.2f s per second of sound'
                      % (mode, fam, f['shown'], f['n'], f['lower'], f['wrong'], f['out_med'], f['out_shown'], f['out_mean_shown'], f['out_worse'], f['onto_med'], f['onto_shown'], f['onto_worse'], f['secs']))
