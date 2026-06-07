import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_and_prep_data():
    print("Fetching UCI Heart Disease Dataset...")

    # Fetch dataset by ID
    heart_disease = fetch_ucirepo(id=45)

    # Access data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine into a single dataframe for easy saving
    df = pd.concat([X, y], axis=1)

    # Drop any rows with missing values to establish a clean baseline
    df = df.dropna()

    # Save to a local CSV for our MLflow pipeline
    df.to_csv("heart_disease_cleaned.csv", index=False)
    print(f"Data saved successfully! Shape: {df.shape}")


if __name__ == "__main__":
    load_and_prep_data()