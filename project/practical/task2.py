# Task 2 – Preprocessing Pipeline + Simple Tests



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split features/target
X = df.drop("income", axis=1)
y = (df["income"] == " >50K").astype(int)

# Ensure y is a pandas Series with the correct data type
y = pd.Series(y, dtype=int)

print(X)
print("-----------------------------------------------------------------------------")
print(y)

# Feature groups
num_feats = X.select_dtypes(include=["int64", "float64"]).columns
cat_feats = X.select_dtypes(include=["object"]).columns

# Preprocessing pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_feats),
    ("cat", categorical_transformer, cat_feats)
])

# Test pipeline
Xt = preprocessor.fit_transform(X)
print("Transformed shape:", Xt.shape)
assert Xt.shape[0] == X.shape[0]
print("✅ Pipeline works correctly")
