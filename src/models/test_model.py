import sys
import yaml
import joblib
import pathlib
import pandas as pd
from dvclive.live import Live
from sklearn.metrics import root_mean_squared_error, r2_score

def test_performance(test_df, model):
    X_test = test_df.drop(columns={"Calories"})
    Y_test = test_df["Calories"]

    Y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    return rmse, r2

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir / "params.yaml"
    params = yaml.safe_load(open(params_file))["test_model"]

    input_path = sys.argv[1]
    model_path = sys.argv[2]

    test_df = pd.read_csv(f"{input_path}/test.csv")
    model = joblib.load(model_path)
    rmse, r2 = test_performance(test_df, model)

    # Use Live context manager properly
    with Live() as live:
        live.log_metric("rmse", float(rmse))
        live.log_metric("r2", float(r2))
        live.next_step()

    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

if __name__ == "__main__":
    main()
