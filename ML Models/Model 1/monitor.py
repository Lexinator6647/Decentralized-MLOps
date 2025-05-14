import numpy as np
import time
import joblib

# Load Model_1
model = joblib.load('model_1.pkl')

# Sample inference inputs
sample_inputs = np.random.rand(5, 5)

# Measure inference performance
start = time.time()
predictions = model.predict(sample_inputs)
duration = time.time() - start

print(f'Predictions: {predictions}')
print(f'Inference time for {len(sample_inputs)} samples: {duration:.4f} seconds')