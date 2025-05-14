import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

X = np.random.rand(100, 5)
y = X.sum(axis=1) + np.random.randn(100) * 0.1
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)
joblib.dump(model, 'model_1.pkl')
print('Model_1 trained and saved.')