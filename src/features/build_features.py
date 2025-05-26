import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def apply_log_transformation(df):
    df['Age'] = np.log(df['Age'] + 1)

def apply_square_transformation(df):
    df['Body_Temp'] = np.log(df['Body_Temp'].max() + 1 - df['Body_Temp'])

def main():

    df_train = pd.read_csv("A:\\Aniket_Scidentai\\MLOPS\\predict_calorie_expenditure\\data\\data\\processed\\train.csv.csv")
    df_test = pd.read_csv("A:\\Aniket_Scidentai\\MLOPS\\predict_calorie_expenditure\\data\\data\\processed\\test.csv.csv")

    apply_log_transformation(df_train)
    apply_log_transformation(df_test)

    apply_square_transformation(df_train)
    apply_square_transformation(df_test)
    