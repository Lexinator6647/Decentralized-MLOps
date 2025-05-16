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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
import joblib

from BlockchainWrapper import BlockchainMetricsWrapper

def train_and_eval_model_1():
    # TRAIN
    X = np.random.rand(100, 5)
    y = X.sum(axis=1) + np.random.randn(100) * 0.1
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'model_1.pkl')

    # EVALUATE
    X_test = np.random.rand(20, 5)
    y_true = X_test.sum(axis=1) + np.random.randn(20) * 0.1
    y_pred = model.predict(X_test)

    return {
        'accuracy': accuracy_score((y_true>y_true.mean()).astype(int),
                                   (y_pred>y_true.mean()).astype(int)),
        'precision': precision_score((y_true>y_true.mean()).astype(int),
                                     (y_pred>y_true.mean()).astype(int)),
        'MSE': mean_squared_error(y_true, y_pred)
    }

if __name__ == "__main__":
    # locate ML_Verse folder at project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mlverse_dir   = os.path.join(project_root, "ML_Verse")

    metrics = train_and_eval_model_1()

    wrapper = BlockchainMetricsWrapper(ml_step="train", output_dir=mlverse_dir)
    wrapper.save_metrics(metrics)        # writes ML_Verse/train_metrics.json
    wrapper.send_to_blockchain()         # runs ML_Verse/O2_train.js
