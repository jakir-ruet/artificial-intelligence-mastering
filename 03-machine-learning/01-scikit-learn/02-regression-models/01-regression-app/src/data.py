import pandas as pd

def load_data(path):
    df = pd.read_csv("data/house_data.csv")
    print("Data Loaded:", df.shape)
    return df
