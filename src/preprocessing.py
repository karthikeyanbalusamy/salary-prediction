import pandas as pd
import os

def main():
    df = pd.read_csv("data/raw/salary.csv")

    # Remove the unwanted column : 'Unnamed: 0'
    df = df.drop(columns=['Unnamed: 0'], errors="ignore") 

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    df.to_csv("data/processed/clean.csv", index=False)

    print("Preprocessing completed successfully")

if __name__ == "__main__":
    main()
