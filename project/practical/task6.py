
# Task 6 – Explainability + Robustness


import shap
from sklearn.metrics import accuracy_score

# SHAP explainability
explainer = shap.TreeExplainer(best_rf.named_steps["model"])

# Slice the original X_test DataFrame before preprocessing
X_test_sliced = X_test[:100]

# Transform the sliced DataFrame
X_trans_sliced = best_rf.named_steps["pre"].transform(X_test_sliced)

# Ensure X_trans_sliced is a dense numpy array with a float dtype if it's not already
# (The preprocessor output might be sparse, convert to dense for SHAP)
if hasattr(X_trans_sliced, 'toarray'):
    X_trans_sliced_dense = X_trans_sliced.toarray().astype(float)
else:
    X_trans_sliced_dense = X_trans_sliced.astype(float)


shap_values = explainer.shap_values(X_trans_sliced_dense)

shap.summary_plot(shap_values, X_trans_sliced_dense,
                  feature_names=best_rf.named_steps["pre"].get_feature_names_out())

# Robustness: add noise to age
X_noisy = X_test.copy()
X_noisy["age"] = X_noisy["age"] + np.random.normal(0, 5, len(X_noisy))

acc_clean = accuracy_score(y_test, best_rf.predict(X_test))
acc_noisy = accuracy_score(y_test, best_rf.predict(X_noisy))
print(f"Clean Accuracy: {acc_clean:.3f} | Noisy Accuracy: {acc_noisy:.3f}")

