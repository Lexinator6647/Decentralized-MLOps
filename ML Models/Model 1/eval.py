import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

# Load Model_1
model = joblib.load('model_1.pkl')

# Synthetic test data
X_test = np.random.rand(20, 5)
y_true = X_test.sum(axis=1) + np.random.randn(20) * 0.1

# Predictions
y_pred = model.predict(X_test)

# Regression metric
mse = mean_squared_error(y_true, y_pred)

# Binarize for classification metrics (threshold at mean)
thresh = y_true.mean()
y_true_bin = (y_true > thresh).astype(int)
y_pred_bin = (y_pred > thresh).astype(int)

# Classification metrics
top_acc = accuracy_score(y_true_bin, y_pred_bin)
top_prec = precision_score(y_true_bin, y_pred_bin)

print(f'MSE: {mse:.4f}, Accuracy: {top_acc:.4f}, Precision: {top_prec:.4f}')