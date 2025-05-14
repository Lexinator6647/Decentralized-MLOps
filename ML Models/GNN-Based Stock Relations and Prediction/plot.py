import os
import pickle
import matplotlib.pyplot as plt  
import pandas as pd

from evaluate import evaluate_all_models 
import config

def main():
    # Paths to processed features and targets
    feat_pkl  = os.path.join(config.PROCESSED_DIR, "features.pkl")
    tgt_pkl   = os.path.join(config.PROCESSED_DIR, "targets.pkl")
    graph_dir = config.GRAPHS_DIR

    # Evaluate each model and collect metrics
    results = evaluate_all_models(
        config.MODELS,
        feat_pkl,
        tgt_pkl,
        graph_dir,
        device=None
    )

    # Convert results dict to DataFrame for easy plotting
    df = pd.DataFrame(results).T
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Plot each metric across models
    plt.figure(figsize=(10, 6))
    for m in metrics:
        plt.plot(df.index, df[m], marker="o", label=m)

    # Add titles and labels
    plt.title("Model Performance Comparison")
    plt.xlabel("Model Name")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)  # Metrics range from 0 to 1
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Metrics")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
