import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Synthetic training data
X = np.random.rand(100, 5)
y = X.sum(axis=1) + np.random.randn(100) * 0.1

# Train and save model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)
joblib.dump(model, 'performance_predictor.pkl')
print('Performance Predictor trained and saved.')
