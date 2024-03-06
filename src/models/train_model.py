import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_csv(r"C:\Users\beyon\Documents\DS\DS_project_2\data\processed\df_processed.csv")
df.drop(df.columns[0], axis=1)
target = "Survived"

# --------------------------------------------------------------
# Train test split
# --------------------------------------------------------------

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# --------------------------------------------------------------
# Train model
# --------------------------------------------------------------

# Define preprocessing for numeric columns (scale them)
numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# Define preprocessing for categorical features (encode them)
categorical_features = X.select_dtypes(include=["category"]).columns
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Build the pipeline
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classfier", RandomForestClassifier())]
)

# fit the pipeline to train the model on the training set
model = pipeline.fit(X_train, y_train)

# --------------------------------------------------------------
# Evaluate the model
# --------------------------------------------------------------

# Get predictions
predictions = model.predict(X_test)

# Display metrics
accuracy = accuracy_score(y_test, predictions)


print("Accuracy:", accuracy)

# # Visualize results
# plot_predicted_vs_true(y_test, predictions)
# regression_scatter(y_test, predictions)
# plot_residuals(y_test, predictions, bins=15)

# # --------------------------------------------------------------
# # Export model
# # --------------------------------------------------------------

# ref_cols = list(X.columns)

# """
# In Python, you can use joblib or pickle to serialize (and deserialize) an object structure into (and from) a byte stream. 
# In other words, it's the process of converting a Python object into a byte stream that can be stored in a file.

# https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html

# """

# joblib.dump(value=[model, ref_cols, target], filename="../../models/model.pkl")