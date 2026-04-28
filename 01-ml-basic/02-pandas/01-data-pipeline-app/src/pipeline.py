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
		df = pd.read_csv("data/raw.csv")

		validate(df)

		df = self.cleaner.run(df)
		df = self.transformer.run(df)

		return df
