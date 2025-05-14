import numpy as np
import time
import joblib

# Load model
model = joblib.load('performance_predictor.pkl')

# Sample inference inputs
sample_inputs = np.random.rand(5, 5)

# Measure inference performance
start = time.time()
predictions = model.predict(sample_inputs)
duration = time.time() - start

print(f'Predictions: {predictions}')
print(f'Inference time for {len(sample_inputs)} samples: {duration:.4f} seconds')