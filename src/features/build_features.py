import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("../../data/raw/train.csv")
test = pd.read_csv("../../data/raw/test.csv")
combine = pd.read_csv("../../data/interim/combine_cleaned_new.csv")

# --------------------------------------------------------------
# Feature creation
# --------------------------------------------------------------

# bins = np.linspace(0,80,5)
# group_names = ['0-20','20-40','40-60','60-80']
# combine['Age_binned'] = pd.cut(combine['Age'], bins, labels=group_names)

combine = combine.drop(['Name','Ticket'], axis=1)

combine['Sex'] = combine['Sex'].astype('category') 
combine['Cabin'] = combine['Cabin'].astype('category')
combine['Embarked'] = combine['Embarked'].astype('category')
combine['Title'] = combine['Title'].astype('category')

combine.info()

train = combine[:len(train)]
test = combine[len(train):]



train.to_csv("../../data/processed/train.csv", index=False)
# test_id = pd.read_csv("../../data/raw/test.csv", usecols = ['PassengerId'])
# test['PassengerId'] = test.index

test.to_csv("../../data/processed/test.csv", index=False)
