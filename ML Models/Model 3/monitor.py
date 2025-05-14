import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score

model = joblib.load('model_3.pkl')
records = []
for _ in range(3):
    X_test = np.random.rand(40,4)
    y_true = (X_test.sum(axis=1)+np.random.randn(40)*0.1>2).astype(int)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_true,y_pred)
    acc = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred)
    records.append({'mse': mse, 'accuracy': acc, 'precision': prec})

pair_keys=['run1-2','run1-3','run2-3']
pairs=[(0,1),(0,2),(1,2)]
drift_summary={}
for key,(i,j) in zip(pair_keys,pairs):
    prev,curr = records[i],records[j]
    drift_summary[key]={
        'mse_drift':(curr['mse']-prev['mse'])/prev['mse'] if prev['mse'] else float('nan'),
        'accuracy_drift':(curr['accuracy']-prev['accuracy'])/prev['accuracy'] if prev['accuracy'] else float('nan'),
        'precision_drift':(curr['precision']-prev['precision'])/prev['precision'] if prev['precision'] else float('nan')
    }
avg={m:sum(drift_summary[k][m] for k in pair_keys)/len(pair_keys) for m in ['mse_drift','accuracy_drift','precision_drift']}
print(avg)