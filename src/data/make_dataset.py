import sys
import missingno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------
# Load and clean train data
# --------------------------------------------------------------

train = pd.read_csv("../../data/raw/train.csv")
test = pd.read_csv("../../data/raw/test.csv")

combine = train.append(test, ignore_index=True)
combine.head()

combine.info()
combine.isnull().sum().sort_values(ascending = False)
combine.nunique()
combine.describe()
missingno.matrix(combine)

# create a new feature to extract title names from the Name column
# normalize the titles
# map the normalized titles to the current titles 
# view value counts for the normalized titles
combine['Title'] = combine.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
combine.Title = combine.Title.map(normalized_titles)
print(combine.Title.value_counts())

# group by Sex, Pclass, and Title 
# view the median Age by the grouped features 
# apply the grouped median value on the Age NaN
grouped = combine.groupby(['Sex','Pclass', 'Title'])  
grouped.Age.median()
combine.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))


# fill Cabin NaN with U for unknown
# map first letter of cabin to itself
combine.Cabin = combine.Cabin.fillna('U')
combine.Cabin = combine.Cabin.map(lambda x: x[0])


# fill NaN with most_embarked value
most_embarked = combine.Embarked.value_counts().index[0]
combine.Embarked = combine.Embarked.fillna(most_embarked)

# fill NaN with median fare
combine.Fare = combine.Fare.fillna(combine.Fare.median())

# size of families (including the passenger)
combine['FamilySize'] = combine.Parch + combine.SibSp + 1

# Convert the male and female groups to integer form
combine.Sex = combine.Sex.map({"male": 0, "female":1})

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

combine.to_csv("../../data/interim/combine_cleaned_new.csv", index=False)
