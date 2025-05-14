import numpy as np
import time
import joblib

# Load Model_2
detector = joblib.load('model_2.pkl')

# Sample inputs for monitoring
sample_inputs = np.random.randn(10, 3)

# Measure inference performance
start = time.time()
preds = detector.predict(sample_inputs)
duration = time.time() - start

# Report binary outlier flags
flags = (preds == -1).astype(int)

print(f'Outlier flags: {flags}')
print(f'Inference time: {duration:.4f} seconds')