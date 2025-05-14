import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import config

# Graph Convolutional Network (GCN)
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=config.HID_DIM, out_dim=2):
        super().__init__()
        # Build a list of GCN layers based on NUM_LAYERS
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.drops = torch.nn.ModuleList()
        layers = config.NUM_LAYERS
        # Input layer
        self.convs.append(GCNConv(in_dim, hid_dim))
        # Hidden layers
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hid_dim, hid_dim))
        # Output layer
        self.convs.append(GCNConv(hid_dim, out_dim))
        # BatchNorm and Dropout for hidden layers
        for _ in range(layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hid_dim))
            self.drops.append(torch.nn.Dropout(0.3))

    def forward(self, x, edge_index):
        # Apply each hidden GCN layer with ReLU, batch norm, and dropout
        for conv, bn, drop in zip(self.convs[:-1], self.bns, self.drops):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(drop(x))
        # Final layer (no activation)
        return self.convs[-1](x, edge_index)

# GraphSAGE model: sample & aggregate neighbors
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=config.HID_DIM, out_dim=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.drops = torch.nn.ModuleList()
        layers = config.NUM_LAYERS
        # Input layer
        self.convs.append(SAGEConv(in_dim, hid_dim))
        # Hidden layers
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hid_dim, hid_dim))
        # Output layer
        self.convs.append(SAGEConv(hid_dim, out_dim))
        # BatchNorm and Dropout for hidden layers
        for _ in range(layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hid_dim))
            self.drops.append(torch.nn.Dropout(0.3))

    def forward(self, x, edge_index):
        # Apply GraphSAGE layers with ReLU, batch norm, and dropout
        for conv, bn, drop in zip(self.convs[:-1], self.bns, self.drops):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(drop(x))
        # Final layer
        return self.convs[-1](x, edge_index)

# Graph Attention Network (GAT) with attention outputs
class GATWithAtt(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=config.HID_DIM, out_dim=2, heads=config.GAT_HEADS):
        super().__init__()
        # First GAT layer returns attention weights
        self.gat1 = GATConv(in_dim, hid_dim, heads=heads, dropout=0.2)
        self.bn1  = torch.nn.BatchNorm1d(hid_dim * heads)
        self.drop = torch.nn.Dropout(0.3)
        # Second GAT layer for output
        self.gat2 = GATConv(hid_dim * heads, out_dim, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        # First attention layer
        x1, (edge_idx, att1) = self.gat1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(self.drop(self.bn1(x1)))
        # Second attention layer
        x2, (edge_idx, att2) = self.gat2(x1, edge_index, return_attention_weights=True)
        # Return logits and final attention weights
        return x2, att2

# Temporal Graph Attention Network: includes time convolution
class TemporalGAT(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=config.HID_DIM, out_dim=2, heads=config.GAT_HEADS):
        super().__init__()
        W = config.FEATURE_WINDOW
        # Number of features per time step
        self.window    = W
        self.num_feats = in_dim // W
        # 1D convolution over time dimension
        self.tconv     = torch.nn.Conv1d(self.num_feats, 16, kernel_size=3, padding=1)
        self.bn0       = torch.nn.BatchNorm1d(16 * W)
        self.drop0     = torch.nn.Dropout(0.3)
        # First GAT layer
        self.gat1      = GATConv(16 * W, hid_dim, heads=heads, dropout=0.2)
        self.bn1       = torch.nn.BatchNorm1d(hid_dim * heads)
        self.drop1     = torch.nn.Dropout(0.3)
        # Final GAT layer
        self.gat2      = GATConv(hid_dim * heads, out_dim, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        N, _ = x.size()
        # Reshape to (batch, features, window) for temporal conv
        x = x.view(N, self.num_feats, self.window)
        x = F.relu(self.tconv(x))
        x = x.view(N, -1)
        x = self.drop0(self.bn0(x))
        # Apply GAT layers with attention
        h, (edge_idx, att1) = self.gat1(x, edge_index, return_attention_weights=True)
        h = F.elu(self.drop1(self.bn1(h)))
        out, (edge_idx, att2) = self.gat2(h, edge_index, return_attention_weights=True)
        return out, att2

# Factory to get model instance by name
def get_model(name, input_dim):
    return {
        "GCN": lambda: GCN(input_dim),
        "GraphSAGE": lambda: GraphSAGE(input_dim),
        "GATWithAtt": lambda: GATWithAtt(input_dim),
        "TemporalGAT": lambda: TemporalGAT(input_dim)
    }[name]()
