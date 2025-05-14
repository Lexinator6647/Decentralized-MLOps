import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load model
model = joblib.load('performance_predictor.pkl')

# Synthetic test data
X_test = np.random.rand(20, 5)
y_true = X_test.sum(axis=1) + np.random.randn(20) * 0.1

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f'MSE: {mse:.4f}, R2: {r2:.4f}')