import json
import os
import subprocess
from datetime import datetime
from generate_json import JSONSaver

class BlockchainMetricsWrapper:
    VALID_ML_STEPS = ["train", "monitor", "promote"]
    
    METRIC_SCHEMAS = {
        "train": {"accuracy", "precision", "mse", "timestamp"},
        "monitor": {"accuracy_drift", "precision_drift", "mse_drift", "timestamp"},
        "promote": {"model_version", "model_uri", "timestamp"}
    }

    def __init__(self, ml_step: str, output_dir="."):
        if ml_step not in self.VALID_ML_STEPS:
            raise ValueError(f"Invalid step '{ml_step}'. Must be one of: {self.VALID_ML_STEPS}")

        self.ml_step = ml_step
        self.expected_metrics = self.METRIC_SCHEMAS[ml_step]
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.prefix = f"{ml_step}_metrics"
        self.filename = os.path.join(output_dir, f"{self.prefix}.json")
        self.js_update_path = f"MLVerse/{self.ml_step}_policy.js"

    def validate_metrics(self, metrics: dict):
        unexpected_keys = set(metrics.keys()) - self.expected_metrics
        if unexpected_keys:
            raise ValueError(f"Unexpected metrics for '{self.ml_step}': {unexpected_keys}")

    def save_metrics(self, metrics: dict):
        self.validate_metrics(metrics)
        # Use generate JSON utility to parse and save metrics dictionary to a file for API consumption
        saver = JSONSaver(prefix=self.prefix)
        json_str = saver.save(metrics)
        print(f"[{self.ml_step}] Metrics saved to {self.filename}")

    def send_to_blockchain(self): #js file that connects to Forte and submits keys and values for update, customized for train, monitor or promotion
        try:
            result = subprocess.run(
                ['node', js_update_path, self.filename],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[{self.ml_step}] Blockchain update successful:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[{self.ml_step}] Blockchain update failed:\n", e.stderr)

# Example usage
if __name__ == "__main__":
    wrapper = BlockchainMetricsWrapper(ml_step="train")
    wrapper.save_metrics({"accuracy": 0.94, "loss": 0.08})
    wrapper.send_to_blockchain()
