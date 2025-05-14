import numpy as np
import time
import joblib

# Load model
detector = joblib.load('drift_detector.pkl')

# Sample inputs for monitoring
sample_inputs = np.random.randn(10, 3)

# Measure inference performance
start = time.time()
preds = detector.predict(sample_inputs)
duration = time.time() - start

print(f'Inlier/Outlier predictions: {preds}')
print(f'Inference time: {duration:.4f} seconds')