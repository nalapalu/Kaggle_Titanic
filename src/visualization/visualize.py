import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

combine = pd.read_csv("../../data/interim/combine_cleaned_new.csv")

# --------------------------------------------------------------
# EDA 
#--------------------------------------------------------------

categorical_features = [
    "Sex",
    "Cabin",
    "Embarked"
]
combine[categorical_features] = combine[categorical_features].astype("category")
combine.info()

# Correlation matrix

plt.subplots(figsize = (10, 10))
sns.heatmap(combine.corr(), annot=True, fmt='.1%', square=True, cbar=False, annot_kws={'size': '8'})
plt.rcParams['font.size'] = '16'
plt.show()
sns.reset_orig()

# Histogram of distributions

sns.set_style("darkgrid")
numerical_columns = combine.select_dtypes(include=["int64", "float64"]).columns
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
	plt.subplot(len(numerical_columns), 2, idx)
	sns.histplot(combine[feature], kde=True)
  plt.title(f"{feature} | Skewness: {round(combine[feature].skew(), 2)}")
plt.tight_layout()
plt.show()

# # Barplots
sns.barplot(x = 'Sex', y = 'Survived', data = combine)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Gender')
plt.ylim(0,1)
plt.show()

# Mean of survival by Parch
sns.barplot(x = 'Parch', y ='Survived', data = train)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Parch')

# Mean of survival by Pclass
sns.barplot(x = 'Pclass', y ='Survived', data = train)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Pclass')
plt.ylim(0,1)
plt.show()


# Mean of survival by Age
bins = np.linspace(0,80,5)
# bins = np.array([18,25,35,45,55,65])
group_names = ['0-20','20-40','40-60','60-80']
train['Age_binned'] = pd.cut(train['Age'], bins, labels=group_names)
sns.barplot(y="Survived",x="Age_binned",data=train)
plt.xlabel("Age",fontsize=20)
plt.ylabel("Survival probability",fontsize=20)
plt.ylim(0,1)
plt.show()

# Mean of survival by Family total
train['FamTot'] = train['SibSp'] + train['Parch'] + 1
sns.barplot(y="Survived",x="FamTot",data=train)
plt.xlabel("Family total",fontsize=20)
plt.ylabel("Survival probability",fontsize=20)
plt.ylim(0,1)
plt.show()

