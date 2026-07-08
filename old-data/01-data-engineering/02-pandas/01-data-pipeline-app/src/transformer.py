import pandas as pd

from sklearn.preprocessing import StandardScaler

class DataTransformer:

	def __init__(self):
		self.scaler = StandardScaler()

	def feature_engineering(self, df):
		df['salary'] = df['salary'] / 1000
		df['age_group'] = df['age'].apply(
			lambda x: 'Young' if x < 30 else 'Adult'
		)
		return df

	def date_features(self, df):
		df['year'] = df['date'].dt.year
		df['month'] = df['date'].dt.month
		return df

	def encode(self, df):
		df = pd.get_dummies(df, columns=['department'])
		return df

	def scale(self, df):
		print(type(self.scaler))
		df[['age', 'salary']] = self.scaler.fit_transform(df[['age', 'salary']])
		return df

	def run(self, df):
		df = self.feature_engineering(df)
		df = self.date_features(df)
		df = self.encode(df)
		df = self.scale(df)
		return df
