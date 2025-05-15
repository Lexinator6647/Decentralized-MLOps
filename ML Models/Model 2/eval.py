import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
from datetime import datetime

detector = joblib.load('model_2.pkl')
X_test = np.vstack([np.random.randn(50, 3), np.random.randn(5, 3) + 5])
y_true = np.array([1]*50 + [-1]*5)
y_pred = detector.predict(X_test)
mse = mean_squared_error(y_true, y_pred)
y_true_bin = (y_true == -1).astype(int)
y_pred_bin = (y_pred == -1).astype(int)
acc = accuracy_score(y_true_bin, y_pred_bin)
prec = precision_score(y_true_bin, y_pred_bin)
eval_results = {
    'mse': mse,
    'accuracy': acc,
    'precision': prec,
    'timestamp': datetime.now().isoformat()
}
print(eval_results)