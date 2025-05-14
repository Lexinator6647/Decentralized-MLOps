from data_collection import download_all_data 
from feature_engineering import prepare_features   
from utils import load_train_test_data       
from train import train_model             
import config                             

def main():
    # Step 1: Download all data and save raw files
    download_all_data()
    # Step 2: Prepare features and targets from raw data
    feats, tgts = prepare_features()
    # Step 3: Split data into training and testing based on TEST_START_DATE
    train_list, test_list = load_train_test_data(
        feats, tgts, split_date=config.TEST_START_DATE
    )
    # Step 4: Train each model specified in config.MODELS
    for name in config.MODELS:
        train_model(name, train_list, test_list)

# If this script is run directly, execute the main pipeline
if __name__ == "__main__":
    main()
