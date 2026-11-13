# Writes the fitted rule "is this lag an echo" (fit3.py -> rule3.pkl, a small boosted model) as Source/Core/EchoRule.h,
# and checks the written trees against the model on the fitting data.
#   python3 export_rule.py rule3.pkl ../../Source/Core/EchoRule.h
import sys, pickle, numpy as np
m = pickle.load(open(sys.argv[1], 'rb')); out = sys.argv[2]
base = float(np.ravel(m._baseline_prediction)[0]); trees = [it[0].nodes for it in m._predictors]
def num(v): return 'INF' if np.isposinf(v) else '-INF' if np.isneginf(v) else repr(float(v))
lines = []; start = [0]
for t in trees:
    for nd in t:
        leaf = bool(nd['is_leaf'])
        lines.append('    { %d, %s, %s, %d, %d, %s },' % (0 if leaf else int(nd['feature_idx']), 'true' if leaf else 'false', 'true' if nd['missing_go_to_left'] else 'false',
                                                           0 if leaf else int(nd['left']), 0 if leaf else int(nd['right']), num(nd['value'] if leaf else nd['num_threshold'])))
    start.append(start[-1] + len(t))
nf = int(m.n_features_in_)
src = '''#pragma once
// The fitted rule of the echo engine: is a lag that stands out in a sound's ripple picture an echo? %d small trees over
// %d measures (Echo.h, feats()). Written by Research/echo/export_rule.py from the model fitted by Research/echo/fit3.py:
// do not edit by hand.
#include <limits>

namespace dr
{
namespace echo
{
namespace rule
{
struct Node { int feat; bool leaf, missLeft; int left, right; double v; };      // leaf: v is its value; else: v is the threshold (<= goes left)
constexpr int features = %d, trees = %d;
constexpr double base = %s;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr int start[trees + 1] = { %s };
constexpr Node nodes[] = {
%s
};
} // namespace rule
} // namespace echo
} // namespace dr
''' % (len(trees), nf, nf, len(trees), repr(base), ', '.join(str(s) for s in start), '\n'.join(lines))
open(out, 'w').write(src)
# the same walk as the app's, against the model
def prob(x):
    s = base
    for t in trees:
        i = 0
        while not t[i]['is_leaf']:
            v = x[t[i]['feature_idx']]
            i = (t[i]['left'] if t[i]['missing_go_to_left'] else t[i]['right']) if np.isnan(v) else (t[i]['left'] if v <= t[i]['num_threshold'] else t[i]['right'])
        s += t[i]['value']
    return 1.0 / (1.0 + np.exp(-s))
if len(sys.argv) > 3:
    X = np.load(sys.argv[3]); p = m.predict_proba(X)[:, 1]; q = np.array([prob(x) for x in X]); print('%d nodes in %d trees; largest difference to the model on %d rows: %.2e' % (start[-1], len(trees), len(X), np.abs(p - q).max()))
else: print('%d nodes in %d trees' % (start[-1], len(trees)))
