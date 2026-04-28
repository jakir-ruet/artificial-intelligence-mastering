from src.pipeline import DataPipeline

if __name__ == "__main__":
	pipeline = DataPipeline()

	clean_df, ml_df = pipeline.run("data/raw_data.csv")

	# save clean data
	clean_df.to_csv("data/clean_data.csv", index=False)
	print("clean_data.csv created")

	# save ml ready data
	ml_df.to_csv("data/ml_ready_data.csv", index=False)
	print("ml_ready_data.csv created")
