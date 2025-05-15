import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
from datetime import datetime

def monitor_model_1():
    model = joblib.load('model_1.pkl')
    records = []
    for _ in range(3):
        X_test = np.random.rand(20, 5)
        y_true = X_test.sum(axis=1) + np.random.randn(20) * 0.1
        thresh = y_true.mean()
        y_true_bin = (y_true > thresh).astype(int)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_true, y_pred)
        acc = accuracy_score(y_true_bin, (y_pred > thresh).astype(int))
        prec = precision_score(y_true_bin, (y_pred > thresh).astype(int))
        records.append({'mse': mse, 'accuracy': acc, 'precision': prec})
    pairs = [(0,1), (0,2), (1,2)]
    avg = {'mse_drift': 0.0, 'accuracy_drift': 0.0, 'precision_drift': 0.0}
    for i,j in pairs:
        prev, curr = records[i], records[j]
        avg['mse_drift'] += (curr['mse'] - prev['mse']) / prev['mse'] if prev['mse'] else 0
        avg['accuracy_drift'] += (curr['accuracy'] - prev['accuracy']) / prev['accuracy'] if prev['accuracy'] else 0
        avg['precision_drift'] += (curr['precision'] - prev['precision']) / prev['precision'] if prev['precision'] else 0
    for key in avg:
        avg[key] /= len(pairs)
    avg['timestamp'] = datetime.now().isoformat()
    return avg