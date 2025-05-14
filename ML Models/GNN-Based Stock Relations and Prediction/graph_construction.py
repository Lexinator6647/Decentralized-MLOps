import os
import pandas as pd
import networkx as nx
import config
from datetime import timedelta

def dynamic_correlation_graph(feats, window, threshold):
    # Build a graph based on correlations over the past 'window' days
    # 1. Find date range common to all tickers
    sets = [set(df.index) for df in feats.values()]
    if not sets:
        return nx.Graph()
    common_dates = set.intersection(*sets)
    if not common_dates:
        return nx.Graph()

    # 2. Define start and end date for correlation window
    end_date = max(common_dates)
    start_date = end_date - timedelta(days=window)

    # 3. Collect 1-day return series for each ticker in window
    ret_dict = {}
    for t, df in feats.items():
        col = next((c for c in df.columns if c.startswith("returns_1d_t-0")), None)
        if col is None:
            continue
        series = df[col].loc[(df.index >= start_date) & (df.index <= end_date)]
        if not series.empty:
            ret_dict[t] = series

    if not ret_dict:
        return nx.Graph()

    # 4. Build DataFrame and compute correlation matrix
    ret_df = pd.DataFrame(ret_dict).dropna(how="any")
    corr = ret_df.corr()

    # 5. Create graph edges where |corr| >= threshold
    G = nx.Graph()
    G.add_nodes_from(ret_df.columns)
    for i, a in enumerate(corr.index):
        for j, b in enumerate(corr.columns):
            if i < j and abs(corr.iat[i, j]) >= threshold:
                G.add_edge(a, b, weight=float(corr.iat[i, j]))
    return G


def construct_graph(feats):
    # Build two graphs: short-term and long-term correlations
    Gs = dynamic_correlation_graph(feats, config.SHORT_CORR_WINDOW, config.CORRELATION_THRESHOLD)
    Gl = dynamic_correlation_graph(feats, config.LONG_CORR_WINDOW, config.CORRELATION_THRESHOLD)

    # Merge into a single graph with separate weights
    G = nx.Graph()
    G.add_nodes_from(feats.keys())
    # Add short-term edges
    for u, v, d in Gs.edges(data=True):
        G.add_edge(u, v, weight_short=d['weight'])
    # Add long-term edges, merge if edge exists
    for u, v, d in Gl.edges(data=True):
        if G.has_edge(u, v):
            G[u][v]['weight_long'] = d['weight']
        else:
            G.add_edge(u, v, weight_long=d['weight'])

    # Add sector super-nodes and connect each stock to its sector
    for sector, ticks in config.SECTOR_MAP.items():
        G.add_node(sector)
        for t in ticks:
            if t in feats:
                G.add_edge(sector, t, weight_hyper=1.0)

    # Save full edgelist for reference
    os.makedirs(config.GRAPHS_DIR, exist_ok=True)
    edgelist = [
        {'source': u, 'target': v, **attrs}
        for u, v, attrs in G.edges(data=True)
    ]
    pd.DataFrame(edgelist).to_csv(
        os.path.join(config.GRAPHS_DIR, 'full_edgelist.csv'),
        index=False
    )

    return G