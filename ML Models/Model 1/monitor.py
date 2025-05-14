import numpy as np
import time
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

# Load Model_1
model = joblib.load('model_1.pkl')

# Prepare synthetic test set
X_test = np.random.rand(20, 5)
y_true = X_test.sum(axis=1) + np.random.randn(20) * 0.1
thresh = y_true.mean()
y_true_bin = (y_true > thresh).astype(int)

# Monitor 3 runs, recording metrics and latency
results = []
for run in range(3):
    start = time.time()
    y_pred = model.predict(X_test)
    duration = time.time() - start
    mse = mean_squared_error(y_true, y_pred)
    y_pred_bin = (y_pred > thresh).astype(int)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin)
    results.append({'run': run + 1, 'mse': mse, 'accuracy': acc, 'precision': prec, 'duration': duration})

# Print run summaries
for res in results:
    print(f"Run {res['run']}: MSE={res['mse']:.4f}, Accuracy={res['accuracy']:.4f}, Precision={res['precision']:.4f}, Duration={res['duration']:.4f}s")