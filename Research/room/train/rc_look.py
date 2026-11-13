# The first lags of the room's energy response: read off the click, fitted from the true parts, fitted from the true gains.
import sys, os
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, common as cm, truth, gbfeat, roomcap as rc
def click_W(c):
    a_, b_, pk_ = truth.direct_window(c['h']); hs = c['h'][0] + c['h'][1]; nn = rc.K * cm.HOP + 2 * cm.NF; dd = np.zeros((nn, 2)); dd[:b_ - a_] = hs[a_:b_]; tt = hs[:nn].copy(); tt[:b_] = 0.0; tt = np.vstack([tt, np.zeros((max(0, nn - len(tt)), 2))])
    if a_ > 0: tt = tt[a_:]; tt = np.vstack([tt, np.zeros((nn - len(tt), 2))])
    Pd = cm.bandpow(cm.fwd(dd)); Pt = cm.bandpow(cm.fwd(tt)); c0 = int(np.argmax(Pd.sum(1))); Wc = (Pt[c0:c0 + rc.K] / np.maximum(Pd.sum(0), 1e-30)[None]).T; return np.pad(Wc, ((0, 0), (0, rc.K - Wc.shape[1])))
if __name__ == '__main__':
    for nm in sys.argv[1:]:
        d = '/tmp/claude-0/v2/bench/sets/reverb/%s/' % nm; c = truth.load(d, 'A'); dA, rA = truth.parts(c); ft = cm.features(c['wet'], dA); Wc = click_W(c)
        W0 = rc.energy_response(10 ** (ft['LS'] / 10), 10 ** (ft['LR'] / 10), c1=0.0, tie=0, iters=4); g = gbfeat.target(ft['LE'], ft['LS'])[0]; S, R, va, bo = rc.tracks(ft['LE'], g); W1 = rc.energy_response(S, R, tie=2); W2 = rc.energy_response(S, R, valid=va, band_ok=bo)
        mid = (cm.FC41 > 450) & (cm.FC41 < 7000); f = lambda W: ' '.join('%6.1f' % v for v in 10 * np.log10(W[mid].mean(0)[[0, 1, 2, 3, 4, 5, 6, 8, 11, 15, 20, 30, 45, 70, 100, 150]] + 1e-9))
        print(nm); print('  lag (frames)        ' + ' '.join('%6d' % v for v in [0, 1, 2, 3, 4, 5, 6, 8, 11, 15, 20, 30, 45, 70, 100, 150]))
        for lab, W in (('click', Wc), ('true parts', W0), ('true gains, tie 2', W1), ('true gains, soft', W2)): print('  %-18s  ' % lab + f(W) + '   sum %6.1f dB' % (10 * np.log10(W[mid].sum(1).mean() + 1e-9)))
