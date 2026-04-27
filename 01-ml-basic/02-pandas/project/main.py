from src.ingest import load_data
from src.clean import clean_data
from src.transform import transform_data
from src.analyze import analyze_data

def main():
    df = load_data("data/raw.csv")
    df = clean_data(df)
    df = transform_data(df)
    analyze_data(df)

    df.to_csv("data/processed.csv", index=False)
    print("Pipeline completed!")

if __name__ == "__main__":
    main()
