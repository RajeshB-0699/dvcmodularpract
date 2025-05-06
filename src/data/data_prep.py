import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data from {filepath}")

def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column].fillna(median_value)
        return df
    except Exception as e:
        raise Exception(f"Error in filling missing with median value : {e}")

def save_file(df: pd.DataFrame, filepath : str) -> None:
    try:
        df.to_csv(filepath, index= False)
    except Exception as e:
        raise Exception(f"Error in savinf file to the {filepath} : {e}")

def main():
    try:
        train_data_path = './data/raw/train.csv'
        test_data_path = './data/raw/test.csv'

        processed_datapath = os.path.join("data","processed")
        os.makedirs(processed_datapath)

        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        train_processed_data = fill_missing_with_median(train_data)
        test_processed_data = fill_missing_with_median(test_data)

        save_file(train_processed_data, os.path.join(processed_datapath, "train_processed.csv"))
        save_file(test_processed_data, os.path.join(processed_datapath, "test_processed.csv"))

    except Exception as e:
        raise Exception(f"Error in data_preparation : {e}")
    
if __name__ == "__main__":
    main()


    
 
