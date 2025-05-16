import os
import json
import subprocess
from datetime import datetime

class BlockchainMetricsWrapper:
    def __init__(self, ml_step: str, output_dir: str = ".", schema_path: str = 'Metrics_Schema.json'):
        # Determine schema file path
        base_dir = os.path.dirname(__file__)
        schema_file = schema_path if os.path.isabs(schema_path) or os.path.dirname(schema_path) else os.path.join(base_dir, schema_path)
        if not os.path.isfile(schema_file):
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        with open(schema_file, 'r') as sf:
            raw_schemas = json.load(sf)
        self.metric_schemas = {step: set(vals) for step, vals in raw_schemas.items()}

        # Validate the ML step
        if ml_step not in self.metric_schemas:
            raise ValueError(f"Invalid step '{ml_step}'")
        self.ml_step = ml_step

        # Prepare output path
        os.makedirs(output_dir, exist_ok=True)
        self.filename = os.path.join(output_dir, f"{ml_step}_metrics.json")

        # Prepare JS updater paths
        self.js_update_dir = os.path.join(base_dir, 'ML_Verse')
        self.js_update_path = os.path.join(self.js_update_dir, f"O2_{ml_step}.js")

    def validate_metrics(self, metrics: dict):
        keys = set(metrics)
        expected = self.metric_schemas[self.ml_step]
        unexpected = keys - expected
        missing = expected - keys
        if unexpected:
            raise ValueError(f"Unexpected metrics: {unexpected}")
        if missing:
            raise ValueError(f"Missing metrics: {missing}")

    def save_metrics(self, metrics: dict):
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        data = {
            'accuracy':  f"{metrics['accuracy']:.4f}",
            'precision': f"{metrics['precision']:.4f}",
            'MSE':       f"{metrics['MSE']:.4f}",
            'timestamp': ts
        }
        self.validate_metrics(data)
        with open(self.filename, 'w') as f:
            json.dump(data, f)
        print(f"[{self.ml_step}] Metrics saved to {self.filename}")
        return data

    def send_to_blockchain(self):
        if not os.path.isfile(self.js_update_path):
            print(f"[{self.ml_step}] No script found at {self.js_update_path}, skipping blockchain update.")
            return
        cmd = ['node', os.path.basename(self.js_update_path), self.filename]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.js_update_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[{self.ml_step}] Blockchain update successful:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"[{self.ml_step}] Blockchain update failed:\n{e.stderr.strip()}")
        return
