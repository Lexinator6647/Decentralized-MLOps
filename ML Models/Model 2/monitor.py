import numpy as np
import time
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

# Load Model_2
model = joblib.load('model_2.pkl')

# Prepare synthetic test set
X_test = np.vstack([np.random.randn(50, 3), np.random.randn(5, 3) + 5])
y_true = np.array([1]*50 + [-1]*5)
\# Monitor 3 runs
results = []
for run in range(3):
    start = time.time()
    y_pred = model.predict(X_test)
    duration = time.time() - start
    mse = mean_squared_error(y_true, y_pred)
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin)
    results.append({'run': run + 1, 'mse': mse, 'accuracy': acc, 'precision': prec, 'duration': duration})

# Print run summaries
for res in results:
    print(f"Run {res['run']}: MSE={res['mse']:.4f}, Accuracy={res['accuracy']:.4f}, Precision={res['precision']:.4f}, Duration={res['duration']:.4f}s")
