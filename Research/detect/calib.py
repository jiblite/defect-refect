# The score shown for an effect. The trees give a raw output z (log-odds on the training mix). It is mapped as
#   shown = sigmoid(slope * (z - z65) + logit(0.65)),  capped at 0.999,
# where z65 is the fixed cut (so the cut sits at the Sureness slider's default, 65 %) and slope is that of a logistic fit of
# "the effect is there" against z on the out-of-fold scores of the training sounds (one number per effect): moving the slider
# then moves every effect's cut by the same amount of evidence as measured on sounds the trees had not seen.
import numpy as np
def logit(p): p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
def platt(z, y):
    """Logistic fit y ~ sigmoid(a z + b): (a, b)."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000).fit(np.asarray(z, np.float64).reshape(-1, 1), np.asarray(y, int)); return float(m.coef_[0, 0]), float(m.intercept_[0])
def shown(z, z65, slope): return np.minimum(0.999, 1 / (1 + np.exp(-(slope * (np.asarray(z, np.float64) - z65) + np.log(0.65 / 0.35)))))
