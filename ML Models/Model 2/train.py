import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

X = np.random.randn(200, 3)
detector = IsolationForest(contamination=0.1, random_state=42)
detector.fit(X)
joblib.dump(detector, 'model_2.pkl')
print('Model_2 trained and saved.')