# The bend meter's two numbers (the verdict's logit, the squeeze read at the peak) as a rule for what is kept at all:
# how many case sounds, clean sounds and sounds with other effects pass each pair of limits. Kinds only, never names.
import os, pickle, sys
from common import *
import bfeat, bengine, bmeter
M = bmeter.model(sys.argv[1]); cache = pickle.load(open('data/beval_X.pkl', 'rb')); top = lambda S: max(bengine.tidy(S[0], bengine.SPREAD)[-1], bengine.tidy(S[1], bengine.SPREAD)[-1])
groups = {'distortion cases': [], 'saturation cases, true top 6 dB and more': [], 'saturation cases, true top 3 to 6 dB': [], 'saturation cases, true top under 3 dB': [], 'clean corpus': [], 'other effects': []}
for key, val in cache.items():
    if key.endswith('|T') or not isinstance(val, tuple): continue
    X, vp = val; S, logit = M.run(X, 20 * np.log10(vp[0] / vp[1])); t = top(S); p = key.split('|')
    if p[0] == 'clean': groups['clean corpus'].append((logit, t))
    elif p[1] == 'distortion': groups['distortion cases'].append((logit, t))
    elif p[1] == 'saturation':
        T = cache[key + '|T']; tt = top(T); groups['saturation cases, true top %s' % ('6 dB and more' if tt >= 6 else '3 to 6 dB' if tt >= 3 else 'under 3 dB')].append((logit, t))
    else: groups['other effects'].append((logit, t))
print('%-42s %5s |' % ('kept from logit / with at least dB at the peak', 'n'), ' '.join('%7s' % ('%g/%g' % (L, T)) for L in (-6, -4, -2, 0, 1.5, 3) for T in (3, 5)))
for k, v in groups.items():
    v = np.array(v); print('%-42s %5d |' % (k, len(v)), ' '.join('%7d' % ((v[:, 0] >= L) & (v[:, 1] >= T)).sum() for L in (-6, -4, -2, 0, 1.5, 3) for T in (3, 5)))
v = np.array(groups['clean corpus']); print('clean corpus: squeeze read at the peak: median %.2f dB, 90th %.2f, 99th %.2f; logit: median %.1f, 90th %.1f, 99th %.1f' % (np.median(v[:, 1]), np.percentile(v[:, 1], 90), np.percentile(v[:, 1], 99), np.median(v[:, 0]), np.percentile(v[:, 0], 90), np.percentile(v[:, 0], 99)))
