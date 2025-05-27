import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import yaml

def apply_log_transformation(df, a: float):
    if a == 0:
        df['Age'] = np.log(df['Age']  + 1)
    else:
        df['Age'] = np.log(df['Age'] * a + 1)

def apply_square_transformation(df, a: float):
    if a == 0:
        df['Body_Temp'] = np.log(df['Body_Temp'].max() + 1 - df['Body_Temp'])
    else:
        df['Body_Temp'] = np.log(0.65 * (df['Body_Temp'].max() + 1 - 0.65*(df['Body_Temp'])))


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["build_features"]

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(f"{input_path}/train.csv")
    test_df = pd.read_csv(f"{input_path}/test.csv")

    apply_log_transformation(train_df, 0)
    apply_log_transformation(test_df, 0)

    apply_square_transformation(train_df, 0.678)
    apply_square_transformation(test_df, 0.678)

    train_df.to_csv(f"{output_path}/train.csv", index=False)
    test_df.to_csv(f"{output_path}/test.csv", index=False)

if __name__ == "__main__":
    main()