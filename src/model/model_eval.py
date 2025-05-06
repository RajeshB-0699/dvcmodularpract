import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data from filepath : {e}")
    
def prepare_data(data : pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns = ['Potability'])
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error in preparing data : {e}")

def load_model(filepath : str):
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)

        return model
    except Exception as e:
        raise Exception(f"Error in loading model from {filepath} : {e}")
    
def evaluation_model(model, X_test : pd.DataFrame, y_test : pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics_dict = {
            'accuracy': acc,
            'f1_score' : f1,
            'recall_score': recall,
            'precision_score' : precision
        }

        return metrics_dict

    except Exception as e:
        raise Exception(f"Error in getting metrics for the model : {e}")
    
def save_metrics(metrics_dict : dict, filepath : str) -> None:
    try:
        with open(filepath, 'w') as file:
            json.dump(metrics_dict, file, indent = 4)
    except Exception as e:
        raise Exception(f"Error in saving metrics :{e}")
    
def main():
    try:
        test_processed_datapath = "./data/processed/test_processed.csv"
        modelpath = "models/model.pkl"
        metrics_filepath = 'metrics.json'
        test_processed_data = load_data(test_processed_datapath)
        X_test, y_test = prepare_data(test_processed_data)
        loaded_model = load_model(modelpath)
        metrics = evaluation_model(loaded_model, X_test, y_test)
        save_metrics(metrics, metrics_filepath)
    except Exception as e:
        raise Exception(f"Error in model evaluating")
    
if __name__ == "__main__":
    main()