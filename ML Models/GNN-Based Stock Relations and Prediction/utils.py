import os
import torch
import joblib
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import config

# Save a PyTorch model's parameters to a file
def save_model(model, name):
    path = os.path.join(config.MODELS_DIR, f"{name}.pt")
    torch.save(model.state_dict(), path)

# Load a saved model by name
# For GNNs, rebuild the model and load weights; for RandomForest, load joblib file
def load_model(name, in_dim=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if name in ("TemporalGAT", "GATWithAtt", "GCN", "GraphSAGE"):
        from models import get_model
        net = get_model(name, input_dim=in_dim).to(device)
        # Load saved weights
        net.load_state_dict(torch.load(
            os.path.join(config.MODELS_DIR, f"{name}.pt"),
            map_location=device
        ))
        net.eval()  # set to evaluation mode
        return net
    # Otherwise, load a scikit-learn model
    return joblib.load(os.path.join(config.MODELS_DIR, f"{name}.joblib"))

# Calculate evaluation metrics given true labels and predictions
def calculate_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    # If probabilities provided, compute AUC; else set to 0
    roc  = roc_auc_score(y_true, y_prob[:,1]) if y_prob is not None else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

# Prepare graph data objects for training/testing
# feats: dict of feature DataFrames; tgts: dict of target Series
# split_date: date string dividing train and test sets
def load_train_test_data(feats, tgts, split_date=config.TEST_START_DATE):
    from graph_construction import construct_graph
    # Build the combined graph once
    G = construct_graph(feats)

    # Map each node to an index for edge construction
    all_nodes = config.STOCKS + list(config.SECTOR_MAP.keys())
    idx_map = {t: i for i, t in enumerate(all_nodes)}
    src, dst = [], []
    for u, v in G.edges():
        if u in idx_map and v in idx_map:
            src.append(idx_map[u])
            dst.append(idx_map[v])
    # Create edge_index tensor for message passing
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    # Sort dates and split into lists of Data objects
    dates = sorted(next(iter(feats.values())).index)
    train_list, test_list = [], []

    for d in dates:
        X_rows, Y_rows = [], []
        # Gather features and labels for each node at date d
        for t in all_nodes:
            if t in feats and d in feats[t].index:
                X_rows.append(feats[t].loc[d].values)
                Y_rows.append(int(tgts.get(t, pd.Series()).get(d, 0)))
            else:
                # If missing, use zeros
                X_rows.append(np.zeros(len(feats[next(iter(feats))].columns)))
                Y_rows.append(0)
        # Create a PyG Data object for this date's graph
        data = Data(
            x = torch.tensor(np.vstack(X_rows), dtype=torch.float),
            edge_index = edge_index,
            y = torch.tensor(Y_rows, dtype=torch.long)
        )
        # Assign to train or test based on split_date
        if pd.to_datetime(d) < pd.to_datetime(split_date):
            train_list.append(data)
        else:
            test_list.append(data)

    return train_list, test_list