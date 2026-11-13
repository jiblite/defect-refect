# First-pass answers for every training case and scoreboard sound, and the three extra pictures made from them
# (roomnum.second_look), for training the second pass.
import sys, os, glob, time
sys.path.insert(0, '/tmp/claude-0/v2/train')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, torch, net, roomnum
from multiprocessing import Pool
MODEL = os.environ.get('MODEL', 'unet_c'); M = None
def one(f):
    global M
    if M is None:
        torch.set_num_threads(1); ck = torch.load('/tmp/claude-0/v2/train/%s.pt' % MODEL, map_location='cpu', weights_only=False); M = net.UNet(ch=tuple(ck['cfg']['ch'])); M.load_state_dict(ck['state']); M.eval()
    z = np.load(f); LE = z['LE'].astype(np.float32)
    with torch.no_grad(): g = M(torch.from_numpy(net.inputs(LE, z['U64'].astype(np.float32), z['U256'].astype(np.float32)))[None])[0].numpy()
    X = roomnum.second_look(LE.astype(np.float64), g.astype(np.float64)); d, n = os.path.split(f); out = os.path.join(os.path.dirname(d), 'pass2_' + os.path.basename(d), n); np.savez_compressed(out, X=X.astype(np.float16), g1=g.astype(np.float16)); return float(X[0].mean() > -0.99)
if __name__ == '__main__':
    t0 = time.time()
    for lab in os.environ.get('SETS', 'cases,sb').split(','):
        os.makedirs('/tmp/claude-0/v2/train/pass2_%s' % lab, exist_ok=True); fs = sorted(glob.glob('/tmp/claude-0/v2/train/%s/*.npz' % lab))
        with Pool(int(os.environ.get('NP', '2'))) as p: r = p.map(one, fs, chunksize=8)
        print(lab, len(fs), 'room found on %.0f%%' % (100 * np.mean(r)), '%.0f s' % (time.time() - t0), flush=True)
