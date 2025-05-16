import os
import json
import subprocess
from datetime import datetime

# Load metric schemas from external JSON
base_dir = os.path.dirname(__file__)
schema_file = os.path.join(base_dir, 'Metrics_Schema.json')
with open(schema_file, 'r') as sf:
    raw_schemas = json.load(sf)
# Convert lists to sets
METRIC_SCHEMAS = {step: set(vals) for step, vals in raw_schemas.items()}

class BlockchainMetricsWrapper:
    VALID_ML_STEPS = set(METRIC_SCHEMAS.keys())

    def __init__(self, ml_step: str, output_dir: str = "."):
        if ml_step not in self.VALID_ML_STEPS:
            raise ValueError(f"Invalid step '{ml_step}'")
        self.ml_step = ml_step
        os.makedirs(output_dir, exist_ok=True)
        self.filename = os.path.join(output_dir, f"{ml_step}_metrics.json")
        base = os.path.dirname(__file__)
        self.js_update_path = os.path.join(base, "ML_Verse", f"O2_{ml_step}.js")

    def validate_metrics(self, metrics: dict):
        keys = set(metrics)
        expected = METRIC_SCHEMAS[self.ml_step]
        unexpected = keys - expected
        missing = expected - keys
        if unexpected:
            raise ValueError(f"Unexpected metrics: {unexpected}")
        if missing:
            raise ValueError(f"Missing metrics: {missing}")

    def save_metrics(self, metrics: dict):
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        data = {
            'accuracy': f"{metrics['accuracy']:.4f}",
            'precision': f"{metrics['precision']:.4f}",
            'MSE': f"{metrics['MSE']:.4f}",
            'timestamp': ts
        }
        self.validate_metrics(data)
        with open(self.filename, 'w') as f:
            json.dump(data, f)
        print(f"[{self.ml_step}] Metrics saved to {self.filename}")
        return data

    def send_to_blockchain(self):
        # Ensure the script exists
        if not os.path.isfile(self.js_update_path):
            print(f"[{self.ml_step}] No script found at {self.js_update_path}, skipping blockchain update.")
            return

        js_dir = os.path.dirname(self.js_update_path)
        js_file = os.path.basename(self.js_update_path)
        cmd = ['node', js_file, self.filename]
        try:
            result = subprocess.run(
                cmd,
                cwd=js_dir,
                check=False,  # don’t raise; we’ll inspect returncode ourselves
                capture_output=True,
                text=True
            )
            print(f"[{self.ml_step}] Node exit code: {result.returncode}")
            if result.stdout:
                print(f"[{self.ml_step}] Blockchain update stdout:\n{result.stdout.strip()}")
            if result.stderr:
                print(f"[{self.ml_step}] Blockchain update stderr:\n{result.stderr.strip()}")

            if result.returncode == 0:
                print(f"[{self.ml_step}] Blockchain update successful.")
            else:
                print(f"[{self.ml_step}] Blockchain update failed (exit code {result.returncode}).")
        except Exception as e:
            print(f"[{self.ml_step}] Exception running blockchain script: {e}")

