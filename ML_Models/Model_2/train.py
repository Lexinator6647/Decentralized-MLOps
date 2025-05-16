import os
import sys

# allow importing BlockchainWrapper from project root
def add_import_path(levels_up=1):
    p = os.path.abspath(__file__)
    for _ in range(levels_up):
        p = os.path.dirname(p)
    sys.path.insert(0, p)

add_import_path(2)

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
import joblib

from BlockchainWrapper import BlockchainMetricsWrapper

def train_and_eval_model_2():
    # TRAIN
    X = np.random.randn(200, 3)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    joblib.dump(model, 'model_2.pkl')

    # EVALUATE on a mix of normal + anomaly
    X_test = np.vstack([np.random.randn(50, 3), np.random.randn(5, 3) + 5])
    y_true = np.array([1]*50 + [-1]*5)
    y_pred = model.predict(X_test)

    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)

    return {
        'accuracy': accuracy_score(y_true_bin, y_pred_bin),
        'precision': precision_score(y_true_bin, y_pred_bin),
        'MSE': mean_squared_error(y_true, y_pred)
    }

if __name__ == "__main__":
    # locate ML_Verse folder at project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mlverse_dir   = os.path.join(project_root, "ML_Verse")

    metrics = train_and_eval_model_2()

    wrapper = BlockchainMetricsWrapper(ml_step="train", output_dir=mlverse_dir)
    wrapper.save_metrics(metrics)
    wrapper.send_to_blockchain()
