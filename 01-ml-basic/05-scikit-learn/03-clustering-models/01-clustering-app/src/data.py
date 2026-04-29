import pandas as pd

def load_data(path):
    df = pd.read_csv("data/customer_data.csv")
    print("Data Loaded:", df.shape)
    return df
