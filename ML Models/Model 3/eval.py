import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
from datetime import datetime

clf = joblib.load('model_3.pkl')
X_test = np.random.rand(40, 4)
y_true = (X_test.sum(axis=1) + np.random.randn(40) * 0.1 > 2).astype(int)
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
eval_results = {
    'mse': mse,
    'accuracy': acc,
    'precision': prec,
    'timestamp': datetime.now().isoformat()
}
print(eval_results)