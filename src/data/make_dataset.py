import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def make_train_test_data(df):
    train_csv, test_csv = train_test_split(df, test_size=0.2, random_state=42)
    return train_csv, test_csv

def main():

    # Process raw data into train/test splits and save to output path.
    df = pd.read_csv("A:\\Aniket_Scidentai\\MLOPS\\predict_calorie_expenditure\\data\\raw\\Predict_Calorie_Expenditure.csv")

    train_csv, test_csv = make_train_test_data(df)

    # Create output directory if it doesn't exist
    output_dir = Path("A:\\Aniket_Scidentai\\MLOPS\\predict_calorie_expenditure\\data\\processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv.to_csv(output_dir / 'train.csv', index=False)
    test_csv.to_csv(output_dir / 'test.csv', index=False)

if __name__ == '__main__':
    main()
