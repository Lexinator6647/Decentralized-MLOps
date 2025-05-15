import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
from datetime import datetime

model = joblib.load('model_1.pkl')
X_test = np.random.rand(20, 5)
y_true = X_test.sum(axis=1) + np.random.randn(20) * 0.1
y_pred = model.predict(X_test)
mse = mean_squared_error(y_true, y_pred)
thresh = y_true.mean()
y_true_bin = (y_true > thresh).astype(int)
y_pred_bin = (y_pred > thresh).astype(int)
acc = accuracy_score(y_true_bin, y_pred_bin)
prec = precision_score(y_true_bin, y_pred_bin)
eval_results = {
    'mse': mse,
    'accuracy': acc,
    'precision': prec,
    'timestamp': datetime.now().isoformat()
}
print(eval_results)