import pandas as pd
from src.validator import validate
from src.cleaner import DataCleaner
from src.transformer import DataTransformer
from src.config import CONFIG

class DataPipeline:

	def __init__(self):
		self.cleaner = DataCleaner(CONFIG)
		self.transformer = DataTransformer()

	def run(self, path):
		df = pd.read_csv("data/raw_data.csv")

		validate(df)

		# create clean copy
		clean_df = self.cleaner.run(df.copy())

		# create ml_ready copy
		transformed_df = self.transformer.run(clean_df.copy())

		return clean_df, transformed_df
