import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df_1 = pd.read_csv("../../data/raw/train.csv")
# df.info()
# df.isnull().sum()
# df.nunique()
# df.describe()

is_embarked = df_1['Embarked'].value_counts().idxmax()
df_1["Embarked"].replace(np.nan, is_embarked, inplace=True)

mean_age = df_1['Age'].astype('float').mean(axis=0)
df_1["Age"].replace(np.nan, mean_age, inplace=True)
df_1.isnull().sum()

# df_1['Cabin'].unique()
df_1["Cabin"].replace(np.nan,'NC', inplace=True)

df = df_1.drop(['Name','Ticket'], axis=1)
# df.value_counts()
# df.head()

categorical_features = [
    "Sex",
    "Cabin",
    "Embarked"
]
df[categorical_features] = df[categorical_features].astype("category")
df.info()


# --------------------------------------------------------------
# EDA
# --------------------------------------------------------------

# df["Sex"].replace('male', 1, inplace=True)
# df["Sex"].replace('female', 0, inplace=True)
# df[["Sex"]] = df[["Sex"]].astype("int64")

# plt.subplots(figsize = (10, 10))
# sns.heatmap(df.corr(), annot=True, fmt='.1%', square=True, cbar=False, annot_kws={'size': '8'})
# plt.rcParams['font.size'] = '16'
# plt.show()
# sns.reset_orig()

# sns.set_style("darkgrid")
# numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
# plt.figure(figsize=(14, len(numerical_columns) * 3))
# for idx, feature in enumerate(numerical_columns, 1):
# 	plt.subplot(len(numerical_columns), 2, idx)
# 	sns.histplot(df[feature])
# 	plt.title(f"{feature} ")
# plt.tight_layout()
# plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df.to_csv("../../data/processed/df_processed.csv")