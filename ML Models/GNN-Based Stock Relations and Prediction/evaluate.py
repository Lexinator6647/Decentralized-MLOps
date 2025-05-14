import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from utils import load_train_test_data, calculate_metrics, load_model
import config


def evaluate_all_models(model_list, features_pkl, targets_pkl, graph_dir, device=None):
    # Load features and targets from pickle files
    with open(features_pkl, "rb") as f:
        feats = pickle.load(f)
    with open(targets_pkl, "rb") as f:
        tgts = pickle.load(f)

    # Split data into training and testing based on date
    train_data, test_data = load_train_test_data(
        feats, tgts, split_date=config.TEST_START_DATE
    )

    results = {}
    for name in model_list:
        print(f"Evaluating {name} …")
        if name == "RandomForest":
            # Load the saved Random Forest and predict
            import joblib
            rf = joblib.load(os.path.join(config.MODELS_DIR, f"{name}.joblib"))
            X = np.vstack([d.x.cpu().numpy() for d in test_data])
            y = np.hstack([d.y.cpu().numpy() for d in test_data])
            prob = rf.predict_proba(X)
            pred = prob.argmax(axis=1)
            # Compute metrics (accuracy, F1, AUC, etc.)
            m = calculate_metrics(y, pred, prob)
        else:
            # Load a GNN model and evaluate on test batches
            mdl = load_model(name, in_dim=test_data[0].x.shape[1], device=device)
            from torch_geometric.loader import DataLoader

            y_true, y_pred, y_prob = [], [], []
            for batch in DataLoader(test_data, batch_size=config.BATCH_SIZE):
                batch = batch.to(device)
                out = mdl(batch.x, batch.edge_index)
                # Some GNNs return (logits, attention)
                logits = out[0] if isinstance(out, tuple) else out
                # Convert logits to probabilities
                probs = (
                    F.softmax(logits, dim=1)
                     .detach()
                     .cpu()
                     .numpy()
                )
                # Collect true labels, predictions, and probabilities
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(probs.argmax(axis=1).tolist())
                y_prob.extend(probs.tolist())

            # Compute metrics for this GNN
            m = calculate_metrics(y_true, y_pred, np.array(y_prob))

        # Store metrics for this model
        results[name] = m

    return results
