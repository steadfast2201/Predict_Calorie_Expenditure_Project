import sys
import yaml
import joblib
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

def prepare_dataset_and_train(train_df, n_estimators, max_depth, seed, learning_rate):

    X_train = train_df.drop(columns={"Calories"})
    Y_train = train_df["Calories"]

    # Initialize XGBRegressor with a comprehensive set of hyperparameters
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        objective="reg:squarederror"  # Recommended for regression
    )

    # Fit the model
    model.fit(X_train, Y_train)
    return model

def save_model(model, output_path):
    joblib.dump(model, output_path + '/model.joblib')

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["train_model"]

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    train_df = pd.read_csv(f"{input_file}/train.csv")

    model = prepare_dataset_and_train(
        train_df,
        params["n_estimators"],
        params["max_depth"],
        params["seed"],
        params["learning_rate"]
    )

    save_model(model,output_path)

if __name__ == "__main__":
    main()