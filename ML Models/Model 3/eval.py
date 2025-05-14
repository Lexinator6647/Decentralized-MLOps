import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

# Load Model_3
clf = joblib.load('model_3.pkl')

# Synthetic test data
X_test = np.random.rand(40, 4)
y_true = (X_test.sum(axis=1) + np.random.randn(40) * 0.1 > 2).astype(int)

# Predictions
y_pred = clf.predict(X_test)

# MSE for binary labels
mse = mean_squared_error(y_true, y_pred)

# Classification metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)

print(f'MSE: {mse:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}')