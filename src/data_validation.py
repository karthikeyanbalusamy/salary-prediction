import pandas as pd
import sys

def main():
    df = pd.read_csv("data/raw/salary.csv")

    print("Total rows:", len(df))
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check numeric types
    if not pd.api.types.is_numeric_dtype(df["YearsExperience"]):
        print("\n YearsExperience must be numeric")
        sys.exit(1)

    if not pd.api.types.is_numeric_dtype(df["Salary"]):
        print("\n Salary must be numeric")
        sys.exit(1)

    print("\n Data validation passed")

if __name__ == "__main__":
    main()
