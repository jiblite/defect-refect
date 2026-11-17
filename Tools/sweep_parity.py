# The moving-filter engine (flanger) in C++ (Source/Core/Sweep.h, through drsweep) against the Python prototype it was
# ported from (Research/sweep/sweep.py): the same answer, the same numbers behind the decision, the same sounds?
#   python3 Tools/sweep_parity.py path/to/drsweep a.wav [b.wav ...]            every sound: find, take out, put onto the next sound in the list
#   python3 Tools/sweep_parity.py path/to/drsweep --list file.txt               the same, one path per line
# Prints one line per sound and a summary; exit code 1 if an answer ("is there a moving copy") differs.
import sys, os, re, subprocess, tempfile, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'Research', 'sweep'))
import sweep

def rd(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 48000, p
    return x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
def rel_db(x, ref): return 10 * np.log10(max(float(((x - ref) ** 2).sum()), 1e-300) / max(float((ref ** 2).sum()), 1e-300))

def main():
    exe = sys.argv[1]; paths = sys.argv[2:]
    if paths and paths[0] == '--list': paths = [l.strip() for l in open(paths[1]) if l.strip()]
    tmp = tempfile.mkdtemp(dir='/dev/shm' if os.path.isdir('/dev/shm') else None); worst_num = 0.0; worst_d = 0.0; worst_snd = -999.0; agree = 0; nfound = 0
    for i, p in enumerate(paths):
        pb = paths[(i + 1) % len(paths)]; a = rd(p); b = rd(pb); pre = os.path.join(tmp, 'o')
        r = subprocess.run([exe, p, pb, pre], capture_output=True, text=True, check=True)
        cf = int(re.search(r'found (\d)', r.stdout).group(1)); cc = [l.split() for l in r.stdout.split('\n') if l.startswith('cand')]; tm = ' '.join(l.strip() for l in r.stdout.split('\n') if l.startswith('  '))
        info = []; F = sweep.find(a, info=info); agree += int(cf == int(F is not None)); pyc = [(ch, f) for ch, cl in enumerate(info) for f in cl]; dn = 0.0
        if len(pyc) != len(cc): line = ' the two sides follow up a different number of lines (%d and %d)' % (len(pyc), len(cc)); dn = 9.0
        else:
            line = ''
            for (ch, f), l in zip(pyc, cc):
                v = dict(zip(l[3::2], l[4::2]))
                pairs = [('c1first', f['c1_first']), ('move', f['move']), ('seen', f['seen']), ('g', f['g']), ('fb', f['fb']), ('res', f['res']), ('level', f['level']), ('expect', f['expect']), ('dmax', f['dmax'])] + ([('levelB', f['level_b']), ('still', f['static']), ('jumps', f['fast'])] if 'level_b' in f else [])
                dn = max(dn, max(abs(float(v[k]) - x) for k, x in pairs)); dn = max(dn, 9.0 if int(v['accept']) != int(sweep.accept(f)) else 0.0)
        worst_num = max(worst_num, dn); s = '%-44s found: C++ %d Python %d |%s decision numbers differ by at most %.1e' % (os.path.basename(p)[:44], cf, int(F is not None), line, dn)
        if cf and F is not None:
            nfound += 1; n = len(a); T = n // 256 + 1; dc = np.fromfile(pre + '.d.f64').reshape(2, T); dp = np.stack([f['d'] if f is not None else np.zeros(T) for f in F['ch']]); dd = float(np.abs(dc - dp).max()); worst_d = max(worst_d, dd); snd = {}
            for tag, ref in (('out', sweep.apply(a, F, -1.0)), ('half', sweep.apply(a, F, -0.5)), ('onto', sweep.apply(b, F, 1.0)), ('onto2', sweep.apply(b, F, 2.0)), ('fad', sweep.apply(b, F, 1.0, speed=1.5, sweep=0.5, feedback=0.2))):
                snd[tag] = rel_db(np.fromfile(pre + '.%s.f64' % tag).reshape(2, -1).T, ref)
            worst_snd = max(worst_snd, max(snd.values())); s += ' | delay line %.1e samples | sounds %.0f dB' % (dd, max(snd.values()))
        print(s + '   [C++ ' + tm + ']', flush=True)
        for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    print('%d sounds, %d with a moving copy, %d differences in the answers. Largest difference: decision numbers %.1e, delay line %.1e samples, sounds %.0f dB against the sound'
          % (len(paths), nfound, len(paths) - agree, worst_num, worst_d, worst_snd))
    sys.exit(0 if agree == len(paths) else 1)
main()
