import pandas as pd

def load_data(path):
	df = pd.read_csv("data/loan_data.csv")
	print("Loan data loaded", df.shape)
	return df

