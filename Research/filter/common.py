# Common pieces for the study of DISTORTION, SATURATION, EXCITER and FILTER: the cases, the truth, masked names.
import sys, os, json, glob
sys.path.insert(0, '/tmp/claude-0/v2/bench'); sys.path.insert(0, '/tmp/claude-0/v2/proto')
import numpy as np, scipy.signal as ss, soundfile as sf
import truth, score
SR = 48000; SETS = '/tmp/claude-0/v2/bench/sets/'; CORPUS = '/tmp/claude-0/v2/corpus/'
def cases(fam): return sorted(glob.glob(SETS + fam + '/*/'))
def kind(clip):
    """What kind of sound a corpus clip is, never its name: own recordings are only ever called 'own'."""
    return 'own' if 'real' in clip else clip.split('_')[0]
def db(a, b=1.0): return 10 * np.log10(max(float(np.sum(np.square(a))), 1e-30) / max(float(np.sum(np.square(b))) if not np.isscalar(b) else b, 1e-30))
def rms(x): return float(np.sqrt(np.mean(np.square(x)) + 1e-30))
