import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------




train_data = pd.read_csv("../../data/raw/train.csv")
train_data.info()
train_data.isnull().sum()
train_data.describe()
filter1 = train_data.drop(['Name','PassengerId'], axis=1)

filter1.value_counts()
train_data.head()
train_data.drop(['Name'], axis=1)

plt.subplots(figsize = (10, 10))
sns.heatmap(train_data.corr(), annot=True, fmt='.1%', square=True, cbar=False, annot_kws={'size': '8'})
plt.rcParams['font.size'] = '16'
plt.show()
sns.reset_orig()


# # Select relevant features
# relevant_features = [
#     "season",
#     "mnth",
#     "holiday",
#     "weekday",
#     "workingday",
#     "weathersit",
#     "temp",
#     "atemp",
#     "hum",
#     "windspeed",
#     "rentals",
# ]
# bike_data = bike_data[relevant_features]

# # Convert non-numerical columns to dtype category
# categorical_features = [
#     "season",
#     "mnth",
#     "holiday",
#     "weekday",
#     "workingday",
#     "weathersit",
# ]
# bike_data[categorical_features] = bike_data[categorical_features].astype("category")
# bike_data.info()

# # --------------------------------------------------------------
# # Export dataset
# # --------------------------------------------------------------

# bike_data.to_pickle("../../data/processed/bike_data_processed.pkl")