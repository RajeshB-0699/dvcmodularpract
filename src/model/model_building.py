import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yaml
import pickle

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data from {filepath} : {e}")
    
def load_params(filepath : str, par : str, par2 : str) -> int:
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
            return params[par][par2]
    except Exception as e:
        raise Exception(f"Error in loading params from {filepath} : {e}")
    
def prepare_data(data : pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns = ['Potability'], axis = 1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error in preparing data : {e}")
    
def train_model(X: pd.DataFrame, y: pd.Series, n_estimators : int, max_depth : int):
    try:
        clf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
        clf.fit(X,y)
        return clf
    except Exception as e:
        raise Exception(f"Error in training model : {e}")
    
def save_model(model : RandomForestClassifier, filepath : str) -> None:
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model , file)
    except Exception as e:
        raise Exception(f"Error in saving model to {filepath} : {e}")
    
def main():
    try:
        params_path = "params.yaml"
        train_processed_datapath = "./data/processed/train_processed.csv"
        test_processed_datapath = "./data/processed/test_processed.csv"

        train_processed_data = load_data(train_processed_datapath)
        test_processed_data = load_data(test_processed_datapath)

        model_name = "models/model.pkl"
        model_building = "model_building"
        n_est = "n_estimators"
        m_dep = "max_depth"
        estimators = load_params(params_path, model_building, n_est)
        max_depth = load_params(params_path, "model_building",m_dep)
        X_train, y_train = prepare_data(train_processed_data)
        X_test, y_test = prepare_data(test_processed_data)
        rf = train_model(X_train, y_train, estimators, max_depth)
        save_model(rf, model_name)

    except Exception as e:
        raise Exception(f"Error in data preparation : {e}")
    
if __name__=="__main__":
    main()


