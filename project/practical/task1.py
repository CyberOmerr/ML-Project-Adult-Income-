# Task 1 – Load Dataset & Perform EDA

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Download dataset
!wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O adult.data

# Define column names
cols = ["age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"]

# Load data
df = pd.read_csv("adult.data", names=cols, na_values=" ?", skipinitialspace=True)

print("Shape:", df.shape)
print(df.head())

# --- Fix target column encoding ---
df["income"] = df["income"].str.strip()   # remove whitespace
y = (df["income"] == ">50K").astype(int)  # 1 if >50K, else 0
X = df.drop("income", axis=1)

print("Unique labels in y:", np.unique(y))
print("Counts:\n", pd.Series(y).value_counts())

# Plot class distribution
sns.countplot(x="income", data=df)
plt.title("Income Distribution")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Copy dataset for correlation analysis
df_corr = df.copy()

# Encode categorical features temporarily for correlation
le = LabelEncoder()
for col in df_corr.select_dtypes(include=["object"]).columns:
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))

# Compute correlation matrix
corr_matrix = df_corr.corr()

# Plot heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.show()

#  Missing Values Summary
print("Missing Values per Column:\n")
print(df.isnull().sum())

#  Missing Values Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.isnull(), cbar=True, cmap="viridis")  # cleaner without colorbar
plt.title("Missing Values Heatmap", fontsize=14)
plt.show()

