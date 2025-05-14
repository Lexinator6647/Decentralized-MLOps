import numpy as np
import joblib

# Load model
detector = joblib.load('drift_detector.pkl')

# Synthetic test data with some anomalies
X_test = np.vstack([np.random.randn(50, 3), np.random.randn(5, 3) + 5])
labels = detector.predict(X_test)  # 1: inlier, -1: outlier
num_anomalies = (labels == -1).sum()
print(f'Anomalies detected: {num_anomalies} out of {len(X_test)}')