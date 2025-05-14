import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

# Load Model_2
detector = joblib.load('model_2.pkl')

# Synthetic test data with anomalies
X_test = np.vstack([np.random.randn(50, 3), np.random.randn(5, 3) + 5])
# True labels: 1 for inlier, -1 for outlier
y_true = np.array([1]*50 + [-1]*5)

# Predictions
y_pred = detector.predict(X_test)

# MSE between true and predicted labels
mse = mean_squared_error(y_true, y_pred)

# Convert to binary 0/1 for accuracy/precision (outlier as 1)
y_true_bin = (y_true == -1).astype(int)
y_pred_bin = (y_pred == -1).astype(int)

# Classification metrics
acc = accuracy_score(y_true_bin, y_pred_bin)
prec = precision_score(y_true_bin, y_pred_bin)

print(f'MSE: {mse:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}')