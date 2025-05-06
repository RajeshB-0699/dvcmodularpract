import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml
import os



def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading {filepath} : {e}")

def load_params(filepath :str) -> float:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
            return params['data_collection']['test_size']
    except Exception as e:
        raise Exception(f"Error in loading params {filepath} : {e}")
    

    
def split_data(data : pd.DataFrame, test_size : float) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(data, test_size = test_size, random_state = 42)
    except ValueError as e:
        raise ValueError(f"Error in splitting data for data : {e}")

def save_data(data : pd.DataFrame, filepath : str) -> None:
    return data.to_csv(filepath, index = False)

def main():
    try:
        data_filepath = "https://raw.githubusercontent.com/RajeshB-0699/datasets_raw/refs/heads/main/water_potability.csv"
        params_filepath = "params.yaml"
        raw_data_path = os.path.join("data","raw")
        df = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(df, test_size=test_size)
        os.makedirs(raw_data_path)
        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, 'test.csv'))

    except Exception as e:
        raise Exception(f"Error in data collection : {e}")
    
if __name__ == "__main__":
    main()