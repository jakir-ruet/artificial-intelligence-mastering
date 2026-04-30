import pandas as pd

class DataCleaner:
	def __init__(self, config):
		self.config = config

	def clean_types(self, df):
		df['age'] = pd.to_numeric(df['age'], errors='coerce')
		df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
		df['date'] = pd.to_datetime(df['date'], errors='coerce')
		return df

	def fill_age(self, df):
		df['age'] = df['age'].fillna(df['age'].median())
		return df

	def fill_salary(self, df):
		df['salary'] = df.groupby('department')['salary'].transform(lambda x: x.fillna(x.mean()))
		return df

	def fill_date(self, df):
		df['date'] = df['date'].ffill()
		return df

	def run(self, df):
		df = self.clean_types(df)
		df = self.fill_age(df)
		df = self.fill_salary(df)
		df = self.fill_date(df)
		return df
