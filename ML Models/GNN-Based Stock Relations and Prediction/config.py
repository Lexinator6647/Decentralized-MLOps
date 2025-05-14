import os
from datetime import datetime

# Base directory for the project
BASE_DIR       = os.getcwd()
# Directories for raw and processed data, graphs, models, and results
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
GRAPHS_DIR = os.path.join(DATA_DIR, "graphs")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for d in (RAW_DIR, PROCESSED_DIR, GRAPHS_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Date range for data collection
START_DATE      = "2010-01-01"                 # Start date for training data
END_DATE        = datetime.now().strftime("%Y-%m-%d")  # End date = today
TEST_START_DATE = "2018-01-01"                 # Date to begin testing

# Feature window length (how many past days to include)
FEATURE_WINDOW      = 10

# Technical indicators and their parameter windows
TECHNICAL_INDICATORS = {
    "returns_1d":    1,   # 1-day return
    "returns_5d":    5,   # 5-day return
    "ma_5d":         5,   # 5-day moving average
    "ma_10d":       10,   # 10-day moving average
    "ma_20d":       20,   # 20-day moving average
    "volatility_5d": 5,   # 5-day volatility (std of returns)
    "volatility_10d":10,  # 10-day volatility
    "rsi_14d":      14,   # 14-day Relative Strength Index
    "momentum_5d":   5,   # 5-day momentum
    "momentum_10d": 10,   # 10-day momentum
}

# Additional indicator windows
SMA_WINDOWS        = [9, 50, 100]  # Simple moving average windows
MFI_WINDOW         = 14            # Money Flow Index window

# Graph construction parameters
SHORT_CORR_WINDOW    = 20   # Window for short-term correlation edges
LONG_CORR_WINDOW     = 60   # Window for long-term correlation edges
CORRELATION_THRESHOLD= 0.4  # Minimum |correlation| to create an edge

# Map sectors to their stock tickers
SECTOR_MAP = {
    "Technology": ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD"],
    "Finance":    ["JPM","BAC","WFC","GS","V","MA"],
    "Energy":     ["XOM","CVX","COP","EOG","SLB"],
    "Healthcare": ["JNJ","UNH","PFE","MRK","ABT"],
    "Consumer":   ["PG","KO","WMT","HD","COST"],
}

# List of all model names available in the app
MODELS            = ["TemporalGAT","GATWithAtt","GCN","GraphSAGE","RandomForest"]

# Training hyperparameters
BATCH_SIZE        = 16     # Number of daily graphs per batch
LR                = 1e-3   # Learning rate for optimizer
EPOCHS            = 30     # Maximum training epochs
DROP_EDGE_RATE    = 0.2    # Probability to drop edges for regularization
JITTER_STD        = 0.01   # Noise to add to edge weights
LABEL_SMOOTHING   = 0.1    # Smooth labels for classification

# Learning rate scheduler settings (if using cyclic LR)
LR_SCHEDULER      = "cyclic"
CYCLE_BASE_LR     = 1e-5  # Min learning rate
CYCLE_MAX_LR      = 1e-3  # Max learning rate
CYCLE_STEP_SIZE_UP= 200   # Steps to increase LR

# GAT-specific parameters
GAT_HEADS         = 8     # Number of attention heads

# GNN model architecture settings
HID_DIM           = 128   # Hidden dimension size
NUM_LAYERS        = 3     # Number of graph convolution layers

# Flatten sector map to get a list of all stocks
STOCKS = sum(SECTOR_MAP.values(), [])  # Combined list of all tickers
