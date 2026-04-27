import pandas as pd

def load_data(path):
    df = pd.read_csv("data/raw.csv")
    print("Data Loaded")
    return df
