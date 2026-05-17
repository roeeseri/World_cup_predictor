import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_PATH = "../data/raw/results.csv"

df = pd.read_csv(DATA_PATH)

print(df.shape)
print(df.head())

print(df.info())
print(df.isnull().sum())

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("First match:", df["date"].min())
print("Last match:", df["date"].max())
print("Number of teams:", pd.concat([df["home_team"], df["away_team"]]).nunique())
print("Number of tournaments:", df["tournament"].nunique())

df["total_goals"] = df["home_score"] + df["away_score"]
df["goal_diff"] = df["home_score"] - df["away_score"]

print(df[["home_score", "away_score", "total_goals", "goal_diff"]].describe())

df["year"] = df["date"].dt.year
matches_per_year = df.groupby("year").size()

matches_per_year.plot(figsize=(14, 5), title="International Matches Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Matches")
plt.show()
