import os
import json
import subprocess
from datetime import datetime

class BlockchainMetricsWrapper:
    VALID_ML_STEPS = {"train", "monitor", "promote"}
    METRIC_SCHEMAS = {
        "train": {"accuracy", "precision", "MSE", "timestamp"},
        "monitor": {"accuracy_drift", "precision_drift", "mse_drift", "timestamp"},
        "promote": {"model_version", "model_uri", "timestamp"}
    }

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
        expected = self.METRIC_SCHEMAS[self.ml_step]
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
        if not os.path.isfile(self.js_update_path):
            print(f"[{self.ml_step}] No script found at {self.js_update_path}, skipping blockchain update.")
            return
        try:
            result = subprocess.run(
                ['node', self.js_update_path, self.filename],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[{self.ml_step}] Blockchain update successful:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"[{self.ml_step}] Blockchain update failed:\n{e.stderr.strip()}")
        return
