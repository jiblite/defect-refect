# Runners for effects that are NOT this project's own code: LV2 and LADSPA plugins, sox, pedalboard, room simulation.
import numpy as np, soundfile as sf, subprocess, tempfile, os, shutil
SR = 48000
TMP = '/tmp/claude-0/v2/tmp'; os.makedirs(TMP, exist_ok=True)

def _io(x):
    d = tempfile.mkdtemp(dir=TMP); a = os.path.join(d, 'in.wav'); b = os.path.join(d, 'out.wav')
    sf.write(a, x.astype(np.float32), SR, subtype='FLOAT'); return d, a, b

def _back(d, b, n):
    y, sr = sf.read(b, always_2d=True); shutil.rmtree(d, ignore_errors=True)
    assert sr == SR, sr
    y = y.astype(np.float64)
    if y.shape[1] == 1: y = np.repeat(y, 2, 1)
    y = y[:, :2]
    if len(y) < n: y = np.vstack([y, np.zeros((n - len(y), 2))])
    return y[:n]

HOST = '/tmp/claude-0/v2/bench/host/'
def run_lv2(x, uri, params=None):
    """x: (n,2). Runs an LV2 plugin offline through the project's own small host (control values by port symbol)."""
    d, a, b = _io(x); cmd = [HOST + 'lv2host', a, b, uri] + ['%s=%s' % (k, repr(float(v))) for k, v in (params or {}).items()]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(b): shutil.rmtree(d, ignore_errors=True); raise RuntimeError('lv2host failed (%d): %s' % (r.returncode, r.stderr[-400:]))
    return _back(d, b, len(x))

def run_sox(x, fx):
    """x: (n,2). fx: list of sox effect arguments, e.g. ['reverb', '50']."""
    d, a, b = _io(x); cmd = ['sox', '-D', a, '-e', 'floating-point', '-b', '32', b] + [str(v) for v in fx]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(b): shutil.rmtree(d, ignore_errors=True); raise RuntimeError('sox failed: ' + r.stderr[-400:])
    return _back(d, b, len(x))

def run_ladspa(x, lib, label, args=()):
    """Control inputs in port order; None = the plugin's default."""
    d, a, b = _io(x); cmd = [HOST + 'ladspahost', a, b, '/usr/lib/ladspa/' + lib, label] + ['-' if v is None else repr(float(v)) for v in args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(b): shutil.rmtree(d, ignore_errors=True); raise RuntimeError('ladspahost failed (%d): %s' % (r.returncode, r.stderr[-400:]))
    return _back(d, b, len(x))

def impulse(n=SR * 6, at=4800):
    x = np.zeros((n, 2)); x[at] = 1.0; return x
