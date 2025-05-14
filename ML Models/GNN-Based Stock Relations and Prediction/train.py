import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from models import get_model
from utils import calculate_metrics, save_model
import config

# Focal loss to handle class imbalance during training
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    # Compute cross-entropy loss per example
    ce = F.cross_entropy(inputs, targets, reduction='none')
    # pt = probability of the true class
    pt = torch.exp(-ce)
    # Scale loss by alpha * (1-pt)^gamma
    return (alpha * (1 - pt) ** gamma * ce).mean()

# Train a model (GNN or Random Forest) with given training and test data
def train_model(name, train_data, test_data):
    # If using Random Forest, train and save directly
    if name == "RandomForest":
        # Stack features and labels from train_data list
        X = np.vstack([d.x.numpy() for d in train_data])
        y = np.hstack([d.y.numpy() for d in train_data])
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        # Save model to disk
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        from joblib import dump
        dump(rf, os.path.join(config.MODELS_DIR, f"{name}.joblib"))
        print(f"✓ Trained & saved {name}.joblib")
        return

    # For GNN models, set device and initialize model
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = get_model(name, input_dim=train_data[0].x.shape[1]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=config.LR)
    # Scheduler reduces learning rate if accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='max', factor=0.5, patience=5
    )
    loader    = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

    best_acc = 0.0
    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        # Iterate over batches
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            out    = model(batch.x, batch.edge_index)
            # Some models return (logits, attention)
            logits = out[0] if isinstance(out, tuple) else out
            loss   = focal_loss(logits, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        # Evaluate on test_data
        model.eval()
        ys, ps, preds = [], [], []
        with torch.no_grad():
            for d in test_data:
                d      = d.to(device)
                out    = model(d.x, d.edge_index)
                logits = out[0] if isinstance(out, tuple) else out
                prob   = F.softmax(logits, dim=1).cpu().numpy()
                pred   = prob.argmax(axis=1)
                ys.append(d.y.cpu().numpy())
                ps.append(prob)
                preds.append(pred)

        # Combine lists into arrays
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        y_pred = np.concatenate(preds)
        # Calculate evaluation metrics
        m      = calculate_metrics(y_true, y_pred, y_prob)
        avg_loss = total_loss / len(loader)

        # Print progress
        print(
            f"[{name}] Epoch {epoch}  "
            f"loss={avg_loss:.4f}  "
            f"acc={m['accuracy']:.3f}  "
            f"prec={m['precision']:.3f}  "
            f"rec={m['recall']:.3f}  "
            f"f1={m['f1']:.3f}  "
            f"auc={m['roc_auc']:.3f}"
        )
        # Update scheduler based on accuracy
        scheduler.step(m["accuracy"])

        # Save model if improved
        if m["accuracy"] > best_acc + 1e-4:
            best_acc = m["accuracy"]
            save_model(model, name)

    print(f"✓ Training complete. Best accuracy={best_acc:.3f}")
