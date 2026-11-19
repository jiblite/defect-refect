# The scoreboard's summary lines (the table in the top-level README) from famboard result files.
# Cases whose far left is not scored (a STUTTER group shown: out, out_all null) are left out of the far-left lines.
#   python3 famsum.py a.jsonl [b.jsonl ...]
import sys, json, numpy as np
def load(p): return [json.loads(l) for l in open(p)]
def summ(rows):
    oth = [r for r in rows if r['fam'] != 'reverb']
    def worse(k): v = np.array([r[k] for r in rows if r[k] is not None], float); return 100 * (v < 0).mean()
    return dict(n=len(rows), own=sum(bool(r['found']) for r in rows), reverb=sum('reverb' in r['others'] for r in oth), rest=sum(len([o for o in r['others'] if o != 'reverb']) for r in rows),
                none=sum(len(r['others']) == 0 for r in rows), own_worse=worse('out'), all_worse=worse('out_all'), onto_worse=worse('onto_all'), onto_own_worse=worse('onto'),
                out_med=float(np.median([r['out'] for r in rows if r['out'] is not None])), out_mean=float(np.mean([r['out'] for r in rows if r['out'] is not None])), onto_med=float(np.median([r['onto'] for r in rows])), onto_mean=float(np.mean([r['onto'] for r in rows])))
if __name__ == '__main__':
    S = {}
    for p in sys.argv[1:]:
        rows = load(p); s = summ(rows); S[p] = {r['name']: r for r in rows}
        print('%-22s n %3d | own shown %3d | others: REVERB %2d, rest %2d | no other group %3d | own far left worse %4.1f%% | all shown far left worse %4.1f%% | onto second worse %4.1f%% (own %4.1f%%) | own out med %5.1f mean %6.1f | onto med %5.1f mean %6.1f'
              % (p.split('/')[-1], s['n'], s['own'], s['reverb'], s['rest'], s['none'], s['own_worse'], s['all_worse'], s['onto_worse'], s['onto_own_worse'], s['out_med'], s['out_mean'], s['onto_med'], s['onto_mean']))
    ps = sys.argv[1:]
    for a, b in zip(ps[:-1], ps[1:]):
        A, B = S[a], S[b]; both = [n for n in A if n in B]
        ra = {n for n in both if A[n]['fam'] != 'reverb' and 'reverb' in A[n]['others']}; rb = {n for n in both if B[n]['fam'] != 'reverb' and 'reverb' in B[n]['others']}
        print('%s -> %s (%d cases in both): false REVERB gone: %s | new: %s | kept: %s' % (a.split('/')[-1], b.split('/')[-1], len(both), sorted(ra - rb), sorted(rb - ra), sorted(ra & rb)))
        ch = [(n, A[n]['found'], B[n]['found']) for n in both if bool(A[n]['found']) != bool(B[n]['found'])]; print('   own group shown changed:', ch)
        oth = [(n, sorted(A[n]['others']), sorted(B[n]['others'])) for n in both if sorted(o for o in A[n]['others'] if o != 'reverb') != sorted(o for o in B[n]['others'] if o != 'reverb')]; print('   other groups (not REVERB) changed:', oth)
        big = [(n, round(A[n][k], 1), round(B[n][k], 1)) for n in both for k in ('out',) if A[n][k] is not None and B[n][k] is not None and abs(A[n][k] - B[n][k]) > 5]; print('   own-group score moved by more than 5:', big)
