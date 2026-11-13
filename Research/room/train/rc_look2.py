import sys, os
sys.path.insert(0, '/tmp/claude-0/v2/train'); sys.path.insert(0, '/tmp/claude-0/v2/proto'); sys.path.insert(0, '/tmp/claude-0/v2/bench')
import numpy as np, common as cm, truth, gbfeat, roomcap as rc, rc_look
for nm in sys.argv[1:]:
    d = '/tmp/claude-0/v2/bench/sets/reverb/%s/' % nm; c = truth.load(d, 'A'); dA, rA = truth.parts(c); ft = cm.features(c['wet'], dA); Wc = rc_look.click_W(c)
    St = 10 ** (ft['LS'] / 10); Rt = 10 ** (ft['LR'] / 10); W0 = rc.energy_response(St, Rt, c1=0.0, tie=0, iters=4); g = gbfeat.target(ft['LE'], ft['LS'])[0]; S, R, va, bo = rc.tracks(ft['LE'], g); W1 = rc.energy_response(S, R, valid=va, band_ok=bo)
    print(nm); print('  band centre Hz : ' + ' '.join('%5.0f' % f for f in cm.FC41[::2]))
    for lab, W in (('click', Wc), ('true parts', W0), ('true gains', W1)): print('  %-14s : ' % lab + ' '.join('%5.1f' % v for v in 10 * np.log10(W.sum(1) + 1e-9)[::2]))
    print('  %-14s : ' % 'on the sound' + ' '.join('%5.1f' % v for v in 10 * np.log10(Rt.sum(0) / np.maximum(St.sum(0), 1e-30) + 1e-9)[::2]))
    print('  %-14s : ' % 'direct level' + ' '.join('%5.1f' % v for v in 10 * np.log10(St.sum(0) / St.sum(0).max() + 1e-9)[::2]))
    n1 = rc.numbers(W1); n0 = rc.numbers(W0); nc = rc.numbers(Wc)
    for lab, n in (('T60 click', nc), ('T60 true parts', n0), ('T60 true gains', n1)): print('  %-14s : ' % lab + ' '.join('%5.2f' % v for v in n['T60'][::2]))
