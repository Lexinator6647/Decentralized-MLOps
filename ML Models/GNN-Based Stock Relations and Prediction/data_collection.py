import os
import pickle
import pandas as pd
import yfinance as yf
import config

class DataCollector:
    def download_stock_data(self, tickers=None, start_date=None, end_date=None):
        # Use provided tickers or default list
        tickers    = tickers or config.STOCKS
        # Use provided dates or default from config
        start_date = start_date or config.START_DATE
        end_date   = end_date   or config.END_DATE

        # Download data from Yahoo Finance, adjust prices automatically
        df = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )

        # If multiple tickers, split into dict; else wrap single DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            raw_dict = {t: df[t].copy() for t in df.columns.levels[0]}
        else:
            raw_dict = {tickers[0]: df.copy()}

        # Ensure the raw data directory exists and save the data
        os.makedirs(config.RAW_DIR, exist_ok=True)
        out_path = os.path.join(config.RAW_DIR, "stock_data.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(raw_dict, f)
        print(f"Downloaded stock data → {out_path}")
        return raw_dict

# Utility function to download all data at once
def download_all_data():
    dc = DataCollector()
    # Returns a dict of DataFrames for each ticker
    return dc.download_stock_data(), None
