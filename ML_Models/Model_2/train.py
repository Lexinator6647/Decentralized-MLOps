import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
from datetime import datetime
import os
import sys

def add_import_path(levels_up=1):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), *['..'] * levels_up))
    if base not in sys.path:
        sys.path.append(base)

add_import_path(2)
from BlockchainWrapper import BlockchainMetricsWrapper

def run_model_2():
    X = np.random.randn(200, 3)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    X_test = np.vstack([np.random.randn(50, 3), np.random.randn(5, 3) + 5])
    y_true = np.array([1] * 50 + [-1] * 5)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin)
    return {'mse': mse, 'accuracy': acc, 'precision': prec, 'timestamp': datetime.now().isoformat()}

bw = BlockchainMetricsWrapper(ml_step='train')
metrics = run_model_2()
bw.save_metrics(metrics)