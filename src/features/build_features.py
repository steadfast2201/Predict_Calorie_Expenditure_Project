import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import yaml
from dvclive.live import Live

def apply_log_transformation(df, a: float):
    if a == 0:
        df['Age'] = np.log(df['Age'] + 1)
    else:
        df['Age'] = np.log(df['Age'] * a + 1)

def apply_square_transformation(df, a: float):
    if a == 0:
        df['Body_Temp'] = np.log(df['Body_Temp'].max() + 1 - df['Body_Temp'])
    else:
        df['Body_Temp'] = np.log(0.65 * (df['Body_Temp'].max() + 1 - 0.65 * df['Body_Temp']))

def distribution_plot(df, live):
    sns.set_style("whitegrid")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cols_per_row = 4
    rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        sns.histplot(x=df[col], kde=True, bins=30, color='orange', ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')

    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    # Log the figure using DVCLive
    live.log_image("distribution_plot.png", fig)
    plt.close(fig)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir / "params.yaml"
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

    with Live() as live:
        distribution_plot(train_df, live)

    train_df.to_csv(f"{output_path}/train.csv", index=False)
    test_df.to_csv(f"{output_path}/test.csv", index=False)

if __name__ == "__main__":
    main()
