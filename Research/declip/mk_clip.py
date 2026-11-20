# Builds the declipper's own test cases (the family 'clip', clip_fx.py) like the scoreboard's (Research/detect/mkpos.py).
#   DR_SETS=<folder for the cases> DR_CORPUS=<the clean clips> python3 mk_clip.py [seed]
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'detect')); sys.argv, args = sys.argv[:1], sys.argv[1:]
import mkpos, clip_fx
seed = int(args[0]) if args else 21
for name, n in clip_fx.FAMILY:
    for i in range(n):
        try: r = mkpos.case(name, i, 'clip', 0.0, seed=seed)
        except Exception as e: r = 'FAILED ' + str(e)[:300]
        print('clip', name, i, r, flush=True)
