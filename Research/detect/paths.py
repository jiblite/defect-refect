# Where things are. Every place can be set through the environment; the defaults fit a checkout of the repo with the
# working folders next to these scripts. Nothing here is audio: the sounds are made, measured and thrown away.
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
def _dir(var, default): return os.path.join(os.environ.get(var, default), '')
WORK = _dir('DR_DETECT_WORK', os.path.join(HERE, 'work'))                   # made by these scripts: data/feats/*.pkl, data/*.npz, model/, results/, tmp/
RES = _dir('DR_RESOURCES', os.path.join(REPO, 'Resources'))                 # the app's model files: models.bin, models27.bin (read), verify.bin (export_verify.py --install)
PROTO = _dir('DR_PROTO', os.path.join(HERE, 'proto'))                       # the first prototype's Python modules: feat.py, detect.py, echo.py (appdet.py), fxlib.py, fxlib27.py (own_fx.py)
CORPUS = _dir('DR_CORPUS', os.path.join(HERE, 'corpus'))                    # the scoreboard's clean clips: meta.json and dry/<clip>.wav
SETS = _dir('DR_SETS', os.path.join(HERE, 'sets'))                          # the scoreboard's cases, a folder per family (reverb and tone come from Research/room/bench; mkpos.py adds the 17 effect families)
BENCH = _dir('DR_BENCH', os.path.join(REPO, 'Research', 'room', 'bench'))   # zoo_core.py: the hosts that run the outside effects
def make_work():
    """The working folders, made by the scripts that write into them."""
    for d in ('data/feats', 'model', 'results', 'tmp/zoo', 'tmp/extra'): os.makedirs(WORK + d, exist_ok=True)
