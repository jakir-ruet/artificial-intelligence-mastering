import pandas as pd

def load_data(pat):
	df = pd.read_csv("data/housing-raw-data.csv")
	return df

def preprocessor(df):
	df = df.dropna()

	x = df[['area', 'bedrooms', 'age']]
	y = df['price']

	return x, y
