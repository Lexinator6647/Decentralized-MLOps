import os
import sys

def add_import_path(levels_up=1):
    p = os.path.abspath(__file__)
    for _ in range(levels_up):
        p = os.path.dirname(p)
    sys.path.insert(0, p)

add_import_path(2)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
import joblib
from BlockchainWrapper import BlockchainMetricsWrapper

def train_and_eval():
    # TRAIN
    X = np.random.rand(150, 4)
    y = (X.sum(axis=1) + np.random.randn(150) * 0.1 > 2).astype(int)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'model_3.pkl')

    # EVALUATE
    X_test = np.random.rand(40, 4)
    y_true = (X_test.sum(axis=1) + np.random.randn(40) * 0.1 > 2).astype(int)
    y_pred = model.predict(X_test)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred)
    }

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mlverse_dir   = os.path.join(project_root, "ML_Verse")

    metrics = train_and_eval()

    wrapper = BlockchainMetricsWrapper(ml_step="train", output_dir=mlverse_dir)
    wrapper.save_metrics(metrics)
    wrapper.send_to_blockchain()
