import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

sys.path.append("..")
# from utility import plot_settings
# from utility.visualize import plot_predicted_vs_true, regression_scatter, plot_residuals


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df_1 = pd.read_csv("../../data/raw/test.csv")
# df_1.info()
# df_1.isnull().sum()
# df_1.nunique()
# df_1.describe()


mean_age = df_1['Age'].astype('float').mean(axis=0)
df_1["Age"].replace(np.nan, mean_age, inplace=True)

mean_fare = df_1['Fare'].astype('float').mean(axis=0)
df_1["Fare"].replace(np.nan, mean_fare, inplace=True)

df_1["Cabin"].replace(np.nan,'NC', inplace=True)

df = df_1.drop(['Name','Ticket'], axis=1)


# df.value_counts()
# df.head()

# --------------------------------------------------------------
# Load model
# --------------------------------------------------------------

model, ref_cols, target = joblib.load("../../models/model.pkl")

# --------------------------------------------------------------
# Make predictions
# --------------------------------------------------------------

X_new = df[ref_cols]
predictions = model.predict(X_new)

results_df = pd.DataFrame({
    'PassengerId': df_1['PassengerId'],
    'Survived': predictions
})


results_df.to_csv('../../data/processed/chiran_submission.csv', index=False)



# --------------------------------------------------------------
# Evaluate results
# --------------------------------------------------------------

# rmse = np.sqrt(mean_squared_error(y_new, predictions))
# r2 = r2_score(y_new, predictions)
# accuracy = accuracy_score(y_new, predictions)

# print("Accuracy:", accuracy)
# print("RMSE:", rmse)
# print("R2:", r2)

# # Visualize results
# plot_predicted_vs_true(y_new, predictions)