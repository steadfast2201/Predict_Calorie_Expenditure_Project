import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    df_train = pd.read_csv("A:\\Aniket_Scidentai\\MLOPS\\predict_calorie_expenditure\\data\\data\\processed\\train.csv.csv")
    df_test = pd.read_csv("A:\\Aniket_Scidentai\\MLOPS\\predict_calorie_expenditure\\data\\data\\processed\\test.csv.csv")

    apply_log_transformation(df_train, 0)
    apply_log_transformation(df_test, 0)

    apply_square_transformation(df_train, 0.678)
    apply_square_transformation(df_test, 0.678)
