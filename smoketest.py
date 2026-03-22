import neurokit2 as nk
import numpy as np
import shap
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV

# neurokit2 + numpy 2.x smoke test
sig = np.random.randn(1000)
_, info = nk.ecg_peaks(sig, sampling_rate=100)
print("neurokit2 OK")

# shap TreeExplainer smoke testg
from sklearn.ensemble import RandomForestClassifier
X = np.random.randn(50, 5)
y = (X[:, 0] > 0).astype(int)
rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
explainer = shap.TreeExplainer(rf, data=X, feature_perturbation='interventional')
vals = explainer.shap_values(X[:5])
print("shap TreeExplainer OK")

print("All compatibility checks passed")