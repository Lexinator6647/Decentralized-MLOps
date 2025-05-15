import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

def run_model_1():
    X = np.random.rand(100, 5)
    y = X.sum(axis=1) + np.random.randn(100) * 0.1
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    X_test = np.random.rand(20, 5)
    y_true = X_test.sum(axis=1) + np.random.randn(20) * 0.1
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    thresh = y_true.mean()
    y_true_bin = (y_true > thresh).astype(int)
    y_pred_bin = (y_pred > thresh).astype(int)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin)
    return {'mse': mse, 'accuracy': acc, 'precision': prec, 'timestamp': datetime.now().strftime("%Y-%m-%d_%H%M%S")}

bw = BlockchainMetricsWrapper(ml_step='train')
metrics = run_model_1()
bw.save_metrics(metrics)


