import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Synthetic data: features and binary retrain-needed label
d = 4
X = np.random.rand(150, d)
y = (X.sum(axis=1) + np.random.randn(150) * 0.1 > 2).astype(int)

# Train and save model
clf = LogisticRegression(solver='liblinear', random_state=42)
clf.fit(X, y)
joblib.dump(clf, 'model_3.pkl')
print('Model_3 trained and saved.')