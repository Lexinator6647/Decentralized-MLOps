import os
import json
import uuid
from datetime import datetime
import sys

def add_import_path(levels_up=1):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), *['..'] * levels_up))
    if base not in sys.path:
        sys.path.append(base)

add_import_path(2)
from BlockchainWrapper import BlockchainMetricsWrapper

STATE_FILE = 'promote_model_3_state.json'
BUCKET = os.getenv('S3_BUCKET', 'ops')
MODEL = 'model_3'

def promote_model_3():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
        state = {'version': 0}
    state['version'] += 1
    version = state['version']
    uid = uuid.uuid4()
    uri = f"https://{BUCKET}.s3.amazonaws.com/models/{MODEL}/{uid}.pkl"
    timestamp = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
    return {'model_version': version, 'model_uri': uri, 'timestamp': timestamp}

bw = BlockchainMetricsWrapper(ml_step='promote')
metrics = promote_model_3()
bw.save_metrics(metrics)