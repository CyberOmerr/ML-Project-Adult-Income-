# Task 3 – Train Models + CV + Hyperparameter Tuning



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import joblib

print("Unique labels in y:", np.unique(y))
print("Counts:\n", pd.Series(y).value_counts())

from sklearn.model_selection import train_test_split

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # ✅ ensures both 0 and 1 are present
)

print("Train class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

# Baseline models
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(random_state=42)
}

best_models = {}
for name, model in models.items():
    clf = Pipeline([("pre", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.3f} | F1: {f1_score(y_test, y_pred):.3f}")
    best_models[name] = clf

# Hyperparameter tuning (RandomForest)
param_dist = {
    "model__n_estimators":[50,100,200],
    "model__max_depth":[5,10,20,None]
}
rf_clf = Pipeline([("pre", preprocessor), ("model", RandomForestClassifier(random_state=42))])
search = RandomizedSearchCV(rf_clf, param_distributions=param_dist, cv=3, n_iter=5, scoring="f1", random_state=42)
search.fit(X_train, y_train)

print("\nBest RF params:", search.best_params_)
best_rf = search.best_estimator_
joblib.dump(best_rf, "adult_rf_model.joblib")

# ROC Curve
y_score = best_rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
plt.plot(fpr, tpr, label="RandomForest")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
