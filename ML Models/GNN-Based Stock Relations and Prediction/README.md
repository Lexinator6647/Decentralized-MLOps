# GNN‑Based Stock Relations & Prediction

In this project, I explore how stocks influence each other and use that information to make better predictions about tomorrow’s prices. I collected daily price and volume data for 29 large U.S. companies and turned each day’s data into a network, where each stock is a point (a “node”) and connections (or “edges”) represent how closely their recent returns move together. By feeding these daily networks into Graph Neural Networks—a type of model that learns from both individual features and the pattern of connections—I teach the model to forecast whether each stock will go up or down the next day. My goal is to show that including these stock‑to‑stock relationships can boost prediction accuracy compared to treating each stock in isolation.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation](#installation)  
4. [Data Collection](#data-collection)  
5. [Feature Engineering](#feature-engineering)  
6. [Graph Construction](#graph-construction)  
7. [Model Training & Evaluation](#model-training--evaluation)  
8. [Web App Demo](#web-app-demo)  
9. [Future Work](#future-work)  
10. [License](#license)  

## Project Overview
I build a daily-updated graph of 29 large‑cap U.S. stocks, where edges represent short‑term (20 d) and long‑term (60 d) return correlations. I train several GNN models (GCN, GraphSAGE, GAT, Temporal‑GAT) plus a Random Forest baseline. My best model, GraphSAGE, reaches **58.7% accuracy** and **F₁ = 0.654**, outperforming the flat baseline by ~2.4%.

## Repository Structure

```bash

├── data/                      # Raw & processed data
│   ├── raw/                   # Downloaded stock data pickles
│   ├── processed/             # Feature & target pickles
│   └── graphs/                # Edge lists CSVs
├── saved\_models/              # Trained model files (.pt, .joblib)
├── results/                   # Evaluation outputs & plots
├── app.py                     # Streamlit web app demo
├── config.py                  # Global settings & parameters
├── data\_collection.py         # Download raw stock data
├── feature\_engineering.py     # Compute technical indicators
├── graph\_construction.py      # Build correlation graphs
├── main.py                    # End-to-end pipeline
├── models.py                  # GNN model definitions
├── train.py                   # Training loop for all models
├── evaluate.py                # Batch evaluation of models
├── utils.py                   # Helpers (save/load, metrics, data split)
└── plot.py                    # Performance comparison plot

```

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/gaganv007/GNN-Based-Stock-Relations-and-Prediction.git
   cd GNN-Based-Stock-Relations-and-Prediction
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data Collection

Download daily OHLC and volume data from Yahoo Finance:

```bash
python -c "from data_collection import download_all_data; download_all_data()"
```

Raw data pickle: `data/raw/stock_data.pkl`

## Feature Engineering

Compute features and labels:

```bash
python -c "from feature_engineering import prepare_features; prepare_features()"
```

Processed pickles in `data/processed/`

## Graph Construction

Built automatically during feature preparation and data loading. To regenerate full edge list:

```bash
python -c "from graph_construction import construct_graph; import pickle, config; feats = pickle.load(open(config.PROCESSED_DIR+'/features.pkl','rb')); construct_graph(feats)"
```

Edge list CSV: `data/graphs/full_edgelist.csv`

## Model Training & Evaluation

Train all models end-to-end:

```bash
python main.py
```

Evaluate and plot performance:

```bash
python plot.py
```

Results saved under `results/`

## Web App Demo

Launch the Streamlit app:

```bash
streamlit run app.py
```

Features:

* Cumulative returns chart
* Next‑day Up/Down prediction + probability
* Top 5 stocks by predicted Up probability
* Attention-based peer influence graph


## Future Work

* Daily automatic graph updates
* Integrate news/sentiment edges
* Multi‑step forecasting (3‑, 5‑day horizons)
* Portfolio optimization using model outputs
* Live dashboard for real‑time predictions

## License

Released under the MIT License. See `LICENSE` for details.

```
```
