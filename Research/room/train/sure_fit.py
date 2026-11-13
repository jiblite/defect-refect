# Fits the "is there a room on this sound" rule on what sure_stats.py collected: among the sounds the engine finds a room on,
# how sure is that (a number 0..1 that the app's Sureness slider compares)?   MODEL=<name> python3 sure_fit.py
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
MODEL = os.environ.get('MODEL', 'unet_b9'); rows = json.load(open('/tmp/claude-0/v2/train/sure_stats_%s.json' % MODEL)); F = [r for r in rows if r['found']]
y = np.array([r['wet'] for r in F]); feats = os.environ.get('FEATS', 'deep,bands,moments,share').split(',')
col = dict(deep=lambda r: np.log10(r['deep'] + 0.003), bands=lambda r: r['evb'] / 41.0, moments=lambda r: np.log10(r['ev'] + 1.0), share=lambda r: r['share'] / 10.0, decay=lambda r: np.log2(r['tmid']), strength=lambda r: r['kmid'] / 10.0)
X = np.array([[col[f](r) for f in feats] for r in F]); w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum()) * len(y)
m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y, sample_weight=w); p = m.predict_proba(X)[:, 1]
pc = cross_val_predict(LogisticRegression(C=1.0, max_iter=1000), X, y, cv=8, method='predict_proba', params=dict(sample_weight=w))[:, 1]
print('model %s | found: %d with reverb, %d without | features %s' % (MODEL, y.sum(), (1 - y).sum(), feats))
print('rule: sureness = 1 / (1 + exp(-(%.3f %s)))' % (m.intercept_[0], ' '.join('%+.3f*%s' % (c, f) for c, f in zip(m.coef_[0], feats))))
allr = rows; wet = np.array([r['wet'] for r in allr]); sure = np.zeros(len(allr)); sure[[i for i, r in enumerate(allr) if r['found']]] = pc
for cut in (0.5, 0.65, 0.8):
    print('  at %.0f%%: shown on %3.0f%% of the sounds with reverb (clearly wet, DRR < 8: %3.0f%%), on %3.0f%% of the dry ones, on %3.0f%% of the tone-only ones' % (100 * cut, 100 * np.mean(sure[wet == 1] > cut), 100 * np.mean([s > cut for s, r in zip(sure, allr) if r['wet'] and r['drr'] < 8]), 100 * np.mean([s > cut for s, r in zip(sure, allr) if r['what'] == 'dry']), 100 * np.mean([s > cut for s, r in zip(sure, allr) if r['what'] == 'tone only'])))
if 'kind' in allr[0]:
    clean = [i for i, r in enumerate(allr) if r['what'] != 'reverb' and r['kind'] in ('speech-synth', 'synthetic', 'sample-notes')]
    print('  surely dry kinds (computer speech, synthetic, single notes), n=%d: shown at 65%% on %.0f%%' % (len(clean), 100 * np.mean(sure[clean] > 0.65)))
