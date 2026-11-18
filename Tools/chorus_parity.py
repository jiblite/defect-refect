# The chorus engine in C++ (Source/Core/Chorus.h, through drchorus) against the Python prototype it was ported from
# (Research/chorus/chorus.py, engine.py): the same peaks, the same voices, the same sounds?
#   python3 Tools/chorus_parity.py path/to/drchorus a.wav b.wav [a2.wav b2.wav ...]
# Every pair: the chorus is found on a, taken out of it, and put onto b. Sounds of up to 12 s (longer ones the app's code
# measures in pieces, which the prototype does not do). Prints the numbers side by side; exit code 1 if an answer (found,
# number of voices, which way round) differs or a sound is further than -100 dB from the prototype's.
import sys, os, json, subprocess, tempfile, time, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'Research', 'chorus')]
import chorus as C, engine as E

def db(a, b): return 10 * np.log10(((a - b) ** 2).mean() / max((b ** 2).mean(), 1e-30) + 1e-30)
def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)

def main():
    exe = sys.argv[1]; paths = sys.argv[2:]; tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); pre = os.path.join(tmp, 'x'); bad = 0; worst = -999.0
    for pa, pb in zip(paths[0::2], paths[1::2]):
        a = rd(pa); b = rd(pb); t0 = time.time(); r = subprocess.run([exe, pa, pb, pre], capture_output=True, text=True); t1 = time.time()
        if r.returncode: print(r.stderr); bad += 1; continue
        J = json.loads(r.stdout.strip().splitlines()[-1]); print('%s: app %.1f s | %s' % (os.path.basename(pa), t1 - t0, ' '.join(r.stderr.split())), flush=True)
        for ch in range(2):
            A, act = C.picture(a[:, ch]); P = C.peaks(A, act); Q = np.fromfile(pre + '.peaks%d.f64' % ch).reshape(-1, 4); del A
            if len(P) == len(Q) and np.all(P[:, 0] == Q[:, 0]): print('  side %d peaks: %d in the same frames | largest differences: delay %.2e stretch %.2e height %.2e' % (ch, len(P), np.abs(P[:, 1] - Q[:, 1]).max(), np.abs(P[:, 2] - Q[:, 2]).max(), np.abs(P[:, 3] - Q[:, 3]).max()), flush=True)
            else: sp = {(int(p[0]), round(p[1], 2)) for p in P}; sq = {(int(p[0]), round(p[1], 2)) for p in Q}; print('  side %d peaks: prototype %d, app %d, in both %d' % (ch, len(P), len(Q), len(sp & sq)), flush=True)
        F = E.find(a); same = bool(F['found']) == bool(J['found']); print('  prototype: found %d sure %.4f | app: found %d sure %.4f' % (F['found'], F['sure'], J['found'], J['sure']))
        for ch in range(2):
            Vp = F['voices'][ch]; Vc = J['pieces'][0]['voices'][ch] if J['pieces'] else []; same = same and len(Vp) == len(Vc) and all(bool(vp.get('ahead')) == bool(vc['ahead']) for vp, vc in zip(Vp, Vc))
            print('  side %d: voices %d / %d' % (ch, len(Vp), len(Vc)))
            for vp, vc in zip(Vp, Vc):
                print('     %s/%s rate %.6f / %.6f  mid %.4f / %.4f  half %.4f / %.4f  on %.3f / %.3f  loud %.4f / %.4f  seen all round %.3f / %.3f  twins %.3f / %.3f  louder than the sound %d / %d' % ('tri' if vp['kind'] else 'sin', 'tri' if vc['kind'] else 'sin', vp['rate'], vc['rate'], vp['mid'], vc['mid'], vp['half'], vc['half'], vp['on'], vc['on'], vp['loud'], vc['loud'], vp['least'], vc['least'], vp['twin'], vc['twin'], bool(vp.get('ahead')), vc['ahead']))
        bad += int(not same)
        if not same: print('  DIFFERENT ANSWERS')
        if not (F['found'] and J['found'] and same): continue
        for tag, ref in (('out', E.apply(a, F, -1.0)), ('half', E.apply(a, F, -0.5)), ('onto', E.apply(b, F, 1.0, own=False)), ('onto2', E.apply(b, F, 2.0, own=False)), ('fast', E.apply(b, F, 1.0, 1.5, 0.5, own=False)), ('slow', E.apply(a, F, 0.0, 0.5, 1.5, own=True))):
            d = db(np.fromfile(pre + '.%s.f64' % tag).reshape(2, len(ref)).T, ref); worst = max(worst, d); print('  %-6s app against prototype: %.1f dB' % (tag, d), flush=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    print('%d pairs, %d with different answers; the sounds differ by at most %.0f dB' % (len(paths) // 2, bad, worst)); sys.exit(1 if bad or worst > -100.0 else 0)
main()
