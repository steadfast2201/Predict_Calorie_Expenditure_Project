# import sys
# import yaml
# import joblib
# import pathlib
# import pandas as pd
# from dvclive.live import Live
# from sklearn.metrics import root_mean_squared_error, r2_score

# def test_performance(test_df, model):
#     X_test = test_df.drop(columns={"Calories"})
#     Y_test = test_df["Calories"]

#     Y_pred = model.predict(X_test)

#     rmse = root_mean_squared_error(Y_test, Y_pred)
#     r2 = r2_score(Y_test, Y_pred)

#     return rmse, r2

# def main():
#     curr_dir = pathlib.Path(__file__)
#     home_dir = curr_dir.parent.parent.parent
#     # params_file = home_dir / "params.yaml"
#     # params = yaml.safe_load(open(params_file))["test_model"]

#     input_path = sys.argv[1]
#     model_path = sys.argv[2]
#     test_df = pd.read_csv(f"{input_path}/test.csv")
#     model = joblib.load(model_path)
#     rmse, r2 = test_performance(test_df, model)

#     # Use Live context manager properly
#     with Live() as live:
#         live.log_metric("rmse", float(rmse))
#         live.log_metric("r2", float(r2))
#         live.next_step()

#     print(f"RMSE: {rmse}")
#     print(f"R2 Score: {r2}")

# if __name__ == "__main__":
#     main()

import sys
import joblib
import pathlib
import numpy as np
import pandas as pd
import xgboost as xgb
from dvclive.live import Live
from sklearn.metrics import root_mean_squared_error, r2_score

def test_performance_per_estimator(test_df, model):
    X_test = test_df.drop(columns={"Calories"})
    Y_test = test_df["Calories"]

    booster = model.get_booster()
    rmse_scores = []
    r2_scores = []

    # Loop through each boosting round
    for i in range(1, model.n_estimators + 1):
        dmatrix = xgb.DMatrix(X_test)
        Y_pred = booster.predict(dmatrix, iteration_range=(0, i))
        rmse = root_mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        rmse_scores.append(float(rmse))
        r2_scores.append(float(r2))

    return rmse_scores, r2_scores

def main():
    import xgboost as xgb  # Required here for xgb.DMatrix
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_path = sys.argv[1]
    model_path = sys.argv[2]
    
    test_df = pd.read_csv(f"{input_path}/test.csv")
    model = joblib.load(model_path)

    rmse_scores, r2_scores = test_performance_per_estimator(test_df, model)

    with Live() as live:
        for i, (rmse, r2) in enumerate(zip(rmse_scores, r2_scores)):
            live.log_metric("rmse", rmse)
            live.log_metric("r2", r2)
            live.next_step()
    
    # Print final scores
    print(f"Final RMSE: {rmse_scores[-1]}")
    print(f"Final R2 Score: {r2_scores[-1]}")

if __name__ == "__main__":
    main()
