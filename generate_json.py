import json
import os
from datetime import datetime

# reusable class that accepts Python dictionary with metrics and values or other key, value pairs to generate and save JSON file for API call to blockchain
class JSONSaver:
    def __init__(self, directory="ML_Verse", prefix="data"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") #example 2025-05-15_101113
        self.filename = os.path.join(directory, f"{prefix}.json")

    def save(self, data: dict) -> str:
        """Save the input dictionary to a new JSON file and return the JSON string."""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary.")

        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=4)

        return json.dumps(data, indent=4)

# Example usage
if __name__ == "__main__":
    saver = JSONSaver()
    json_str = saver.save({"accuracy": "0.8", "precision": "0.75", "recall": "0.9"})
    print("Saved JSON:")
    print(json_str)
