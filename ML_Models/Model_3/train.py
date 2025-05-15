import numpy as np
from sklearn.linear_model import LogisticRegression
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

def run_model_3():
    X = np.random.rand(150, 4)
    y = (X.sum(axis=1) + np.random.randn(150) * 0.1 > 2).astype(int)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)
    X_test = np.random.rand(40, 4)
    y_true = (X_test.sum(axis=1) + np.random.randn(40) * 0.1 > 2).astype(int)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    return {'mse': mse, 'accuracy': acc, 'precision': prec, 'timestamp': datetime.now().isoformat()}

bw = BlockchainMetricsWrapper(ml_step='train')
metrics = run_model_3()
bw.save_metrics(metrics)