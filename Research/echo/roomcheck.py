# Does the room the room engine reports on a sound with an echo survive taking the echo out?
# Per delay sound of the scoreboard: the room engine's verdict on the sound as it is, and on the sound with its echoes out (drecho out).
import sys, os, re, glob, json, subprocess
B = '/tmp/claude-0/drbuild/'; RES = '/home/claude/defect-refect/Resources'; D = '/dev/shm/ep/'; out = open('roomcheck.jsonl', 'a')
done = {json.loads(l)['name'] for l in open('roomcheck.jsonl')}
def room(path):
    r = subprocess.run([B + 'drrender', RES, path, '-', '-'], capture_output=True, text=True); m = re.search(r'room: found=(\d) detector=([0-9.e+-]+) sureness=([0-9.e+-]+)', r.stdout)
    e = re.search(r'echo: sure=([0-9.e+-]+)', r.stdout)
    return dict(found=int(m.group(1)), sure=float(m.group(3)), echo=float(e.group(1))) if m else None
for f in sorted(glob.glob(D + 'dly_*_wet.wav')):
    name = os.path.basename(f)[:-8]
    if name in done: continue
    a = room(f); subprocess.run([B + 'drecho', 'out', f, D + '_rc.wav'], capture_output=True, text=True); b = room(D + '_rc.wav')
    out.write(json.dumps(dict(name=name, wet=a, without=b)) + '\n'); out.flush(); print(name, a, b, flush=True)
