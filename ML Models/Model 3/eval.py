import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load model
clf = joblib.load('model_3.pkl')

# Synthetic test data
X_test = np.random.rand(40, 4)
y_true = (X_test.sum(axis=1) + np.random.randn(40) * 0.1 > 2).astype(int)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
print(f'Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
