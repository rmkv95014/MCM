import pandas as pd

medal_data = pd.read_csv("2025_Problem_C_Data/summerOly_medal_counts.csv")
athlete_data = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")

medal_data["HasMedal"] = medal_data[[
    "Gold", "Silver", "Bronze"]].sum(axis=1) > 0

first_medal_year = medal_data[medal_data["HasMedal"]].groupby("NOC")[
    "Year"].min()
first_medal_df = first_medal_year.reset_index()
first_medal_df.columns = ["NOC", "FirstMedalYear"]
