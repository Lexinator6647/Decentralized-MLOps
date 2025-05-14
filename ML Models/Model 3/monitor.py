import numpy as np
import time
import joblib

# Load model
clf = joblib.load('model_3.pkl')

# Sample inputs for monitoring
sample_inputs = np.random.rand(5, 4)

# Measure inference performance
start = time.time()
preds = clf.predict(sample_inputs)
duration = time.time() - start

print(f'Retraining-needed flags: {preds}')
print(f'Inference time: {duration:.4f} seconds')
