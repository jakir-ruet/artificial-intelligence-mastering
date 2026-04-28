from src.pipeline import DataPipeline

if __name__ == "__main__":
	pipeline = DataPipeline()

	df = pipeline.run("data/raw.csv")

	print(df.head())
	df.to_csv("data/cleaned_transformed.csv", index=False)
